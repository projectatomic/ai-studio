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
import type { ContainerCreateOptions, ImageInfo } from '@podman-desktop/api';
import type { InferenceServerConfig } from '@shared/src/models/InferenceServerConfig';
import { type BetterContainerCreateResult, InferenceProvider } from './InferenceProvider';
import { getModelPropertiesForEnvironment } from '../../utils/modelsUtils';
import { DISABLE_SELINUX_LABEL_SECURITY_OPTION } from '../../utils/utils';
import { LABEL_INFERENCE_SERVER } from '../../utils/inferenceUtils';
import type { TaskRegistry } from '../../registries/TaskRegistry';
import { InferenceType } from '@shared/src/models/IInference';

export const LLAMA_CPP_INFERENCE_IMAGE =
  'ghcr.io/containers/podman-desktop-extension-ai-lab-playground-images/ai-lab-playground-chat:0.4';

export const SECOND: number = 1_000_000_000;

export class LlamaCppPython extends InferenceProvider {
  constructor(taskRegistry: TaskRegistry) {
    super(taskRegistry, InferenceType.LLAMA_CPP, 'LLama-cpp (CPU)');
  }

  dispose() {}

  public enabled = (): boolean => true;

  protected async getContainerCreateOptions(
    config: InferenceServerConfig,
    imageInfo: ImageInfo,
  ): Promise<ContainerCreateOptions> {
    if (config.modelsInfo.length === 0) throw new Error('Need at least one model info to start an inference server.');

    if (config.modelsInfo.length > 1) {
      throw new Error('Currently the inference server does not support multiple models serving.');
    }

    const modelInfo = config.modelsInfo[0];

    if (modelInfo.file === undefined) {
      throw new Error('The model info file provided is undefined');
    }

    const envs: string[] = [`MODEL_PATH=/models/${modelInfo.file.file}`, 'HOST=0.0.0.0', 'PORT=8000'];
    envs.push(...getModelPropertiesForEnvironment(modelInfo));

    return {
      Image: imageInfo.Id,
      Detach: true,
      ExposedPorts: { [`${config.port}`]: {} },
      HostConfig: {
        AutoRemove: false,
        Mounts: [
          {
            Target: '/models',
            Source: modelInfo.file.path,
            Type: 'bind',
          },
        ],
        SecurityOpt: [DISABLE_SELINUX_LABEL_SECURITY_OPTION],
        PortBindings: {
          '8000/tcp': [
            {
              HostPort: `${config.port}`,
            },
          ],
        },
      },
      HealthCheck: {
        // must be the port INSIDE the container not the exposed one
        Test: ['CMD-SHELL', `curl -sSf localhost:8000/docs > /dev/null`],
        Interval: SECOND * 5,
        Retries: 4 * 5,
      },
      Labels: {
        ...config.labels,
        [LABEL_INFERENCE_SERVER]: JSON.stringify(config.modelsInfo.map(model => model.id)),
      },
      Env: envs,
      Cmd: ['--models-path', '/models', '--context-size', '700', '--threads', '4'],
    };
  }

  async perform(config: InferenceServerConfig): Promise<BetterContainerCreateResult> {
    if (!this.enabled()) throw new Error('not enabled');

    // pull the image
    const imageInfo: ImageInfo = await this.pullImage(
      config.providerId,
      config.image ?? LLAMA_CPP_INFERENCE_IMAGE,
      config.labels,
    );

    // Get the container creation options
    const containerCreateOptions: ContainerCreateOptions = await this.getContainerCreateOptions(config, imageInfo);

    // Create the container
    return this.createContainer(imageInfo.engineId, containerCreateOptions, config.labels);
  }
}
