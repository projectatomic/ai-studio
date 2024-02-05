/**********************************************************************
 * Copyright (C) 2024 Red Hat, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ***********************************************************************/

import {
  containerEngine,
  type Webview,
  type ImageInfo,
  type ProviderContainerConnection,
  provider,
} from '@podman-desktop/api';
import type { LocalModelInfo } from '@shared/src/models/ILocalModelInfo';

import path from 'node:path';
import { getFreePort } from '../utils/ports';
import type { QueryState } from '@shared/src/models/IPlaygroundQueryState';
import { MSG_NEW_PLAYGROUND_QUERIES_STATE, MSG_PLAYGROUNDS_STATE_UPDATE } from '@shared/Messages';
import type { PlaygroundState, PlaygroundStatus } from '@shared/src/models/IPlaygroundState';
import type { ContainerRegistry } from '../registries/ContainerRegistry';
import type { PodmanConnection } from './podmanConnection';
import OpenAI from 'openai';

const LABEL_MODEL_ID = 'ai-studio-model-id';
const LABEL_MODEL_PORT = 'ai-studio-model-port';

// TODO: this should not be hardcoded
const PLAYGROUND_IMAGE = 'quay.io/bootsy/playground:v0';

const STARTING_TIME_MAX = 3600 * 1000;

function findFirstProvider(): ProviderContainerConnection | undefined {
  const engines = provider
    .getContainerConnections()
    .filter(connection => connection.connection.type === 'podman')
    .filter(connection => connection.connection.status() === 'started');
  return engines.length > 0 ? engines[0] : undefined;
}

export class PlayGroundManager {
  private queryIdCounter = 0;

  // Dict modelId => state
  private playgrounds: Map<string, PlaygroundState>;
  private queries: Map<number, QueryState>;

  constructor(
    private webview: Webview,
    private containerRegistry: ContainerRegistry,
    private podmanConnection: PodmanConnection,
  ) {
    this.playgrounds = new Map<string, PlaygroundState>();
    this.queries = new Map<number, QueryState>();
  }

  adoptRunningPlaygrounds() {
    this.podmanConnection.startupSubscribe(() => {
      containerEngine
        .listContainers()
        .then(containers => {
          const playgroundContainers = containers.filter(
            c => LABEL_MODEL_ID in c.Labels && LABEL_MODEL_PORT in c.Labels && c.State === 'running',
          );
          for (const containerToAdopt of playgroundContainers) {
            const modelId = containerToAdopt.Labels[LABEL_MODEL_ID];
            if (this.playgrounds.has(modelId)) {
              continue;
            }
            const modelPort = parseInt(containerToAdopt.Labels[LABEL_MODEL_PORT], 10);
            if (isNaN(modelPort)) {
              continue;
            }
            const state: PlaygroundState = {
              modelId,
              status: 'running',
              container: {
                containerId: containerToAdopt.Id,
                engineId: containerToAdopt.engineId,
                port: modelPort,
              },
            };
            this.updatePlaygroundState(modelId, state);
          }
        })
        .catch((err: unknown) => {
          console.error('error during adoption of existing playground containers', err);
        });
    });
  }

  async selectImage(image: string): Promise<ImageInfo | undefined> {
    const images = (await containerEngine.listImages()).filter(im => im.RepoTags?.some(tag => tag === image));
    return images.length > 0 ? images[0] : undefined;
  }

  setPlaygroundStatus(modelId: string, status: PlaygroundStatus): void {
    this.updatePlaygroundState(modelId, {
      modelId: modelId,
      ...(this.playgrounds.get(modelId) || {}),
      status: status,
    });
  }

  updatePlaygroundState(modelId: string, state: PlaygroundState): void {
    this.playgrounds.set(modelId, state);
    this.webview
      .postMessage({
        id: MSG_PLAYGROUNDS_STATE_UPDATE,
        body: this.getPlaygroundsState(),
      })
      .catch((err: unknown) => {
        console.error(`Something went wrong while emitting MSG_PLAYGROUNDS_STATE_UPDATE: ${String(err)}`);
      });
  }

  async startPlayground(modelId: string, modelPath: string): Promise<string> {
    // TODO(feloy) remove previous query from state?
    if (this.playgrounds.has(modelId)) {
      // TODO: check manually if the contains has a matching state
      switch (this.playgrounds.get(modelId).status) {
        case 'running':
          throw new Error('playground is already running');
        case 'starting':
        case 'stopping':
          throw new Error('playground is transitioning');
        case 'error':
        case 'none':
        case 'stopped':
          break;
      }
    }

    this.setPlaygroundStatus(modelId, 'starting');

    const connection = findFirstProvider();
    if (!connection) {
      this.setPlaygroundStatus(modelId, 'error');
      throw new Error('Unable to find an engine to start playground');
    }

    let image = await this.selectImage(PLAYGROUND_IMAGE);
    if (!image) {
      await containerEngine.pullImage(connection.connection, PLAYGROUND_IMAGE, () => {});
      image = await this.selectImage(PLAYGROUND_IMAGE);
      if (!image) {
        this.setPlaygroundStatus(modelId, 'error');
        throw new Error(`Unable to find ${PLAYGROUND_IMAGE} image`);
      }
    }

    const freePort = await getFreePort();
    const result = await containerEngine.createContainer(image.engineId, {
      Image: image.Id,
      Detach: true,
      ExposedPorts: { ['' + freePort]: {} },
      HostConfig: {
        AutoRemove: true,
        Mounts: [
          {
            Target: '/models',
            Source: path.dirname(modelPath),
            Type: 'bind',
          },
        ],
        PortBindings: {
          '8000/tcp': [
            {
              HostPort: '' + freePort,
            },
          ],
        },
      },
      Labels: {
        [LABEL_MODEL_ID]: modelId,
        [LABEL_MODEL_PORT]: `${freePort}`,
      },
      Env: [`MODEL_PATH=/models/${path.basename(modelPath)}`],
      Cmd: ['--models-path', '/models', '--context-size', '700', '--threads', '4'],
    });

    const disposable = this.containerRegistry.subscribe(result.id, (status: string) => {
      switch (status) {
        case 'remove':
        case 'die':
        case 'cleanup':
          // Update the playground state accordingly
          this.updatePlaygroundState(modelId, {
            status: 'none',
            modelId,
          });
          disposable.dispose();
          break;
      }
    });

    let contacted = false;
    const start = Date.now();
    while (
      Date.now() - start < STARTING_TIME_MAX &&
      !contacted &&
      this.playgrounds.get(modelId).status === 'starting'
    ) {
      try {
        const response = await fetch(`http://localhost:${freePort}`);
        contacted = response.ok;
      } catch (err: unknown) {
        /* empty */
      }
    }

    if (!contacted) {
      await containerEngine.stopContainer(image.engineId, result.id);
      throw new Error(`Can't start playground for model ${modelId}`);
    }

    this.updatePlaygroundState(modelId, {
      container: {
        containerId: result.id,
        port: freePort,
        engineId: image.engineId,
      },
      status: 'running',
      modelId,
    });

    return result.id;
  }

  async stopPlayground(modelId: string): Promise<void> {
    const state = this.playgrounds.get(modelId);
    if (state?.container === undefined) {
      throw new Error('model is not running');
    }
    this.setPlaygroundStatus(modelId, 'stopping');
    // We do not await since it can take a lot of time
    containerEngine
      .stopContainer(state.container.engineId, state.container.containerId)
      .then(async () => {
        this.setPlaygroundStatus(modelId, 'stopped');
      })
      .catch(async (error: unknown) => {
        console.error(error);
        this.setPlaygroundStatus(modelId, 'error');
      });
  }

  async askPlayground(modelInfo: LocalModelInfo, prompt: string): Promise<number> {
    const state = this.playgrounds.get(modelInfo.id);
    if (state?.container === undefined) {
      throw new Error('model is not running');
    }

    const query: QueryState = {
      id: this.getNextQueryId(),
      modelId: modelInfo.id,
      prompt: prompt,
    };

    const client = new OpenAI({ baseURL: `http://localhost:${state.container.port}/v1`, apiKey: 'dummy' });

    const response = await client.completions.create({
      model: modelInfo.file,
      prompt,
      temperature: 0.7,
      max_tokens: 1024,
      stream: true,
    });

    this.queries.set(query.id, query);
    this.sendQueriesState();

    (async () => {
      for await (const chunk of response) {
        query.error = undefined;
        if (query.response) {
          query.response.choices = query.response.choices.concat(chunk.choices);
        } else {
          query.response = chunk;
        }
        this.sendQueriesState();
      }
    })().catch((err: unknown) => console.warn(`Error while reading streamed response for model ${modelInfo.id}`, err));

    return query.id;
  }

  getNextQueryId() {
    return ++this.queryIdCounter;
  }
  getQueriesState(): QueryState[] {
    return Array.from(this.queries.values());
  }

  getPlaygroundsState(): PlaygroundState[] {
    return Array.from(this.playgrounds.values());
  }

  sendQueriesState(): void {
    this.webview
      .postMessage({
        id: MSG_NEW_PLAYGROUND_QUERIES_STATE,
        body: this.getQueriesState(),
      })
      .catch((err: unknown) => {
        console.error(`Something went wrong while emitting MSG_NEW_PLAYGROUND_QUERIES_STATE: ${String(err)}`);
      });
  }
}
