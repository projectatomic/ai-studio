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
  type ContainerProviderConnection,
  type ImageInfo,
  type ListImagesOptions,
  provider,
  type ProviderContainerConnection,
  type PullEvent,
} from '@podman-desktop/api';
import type { CreationInferenceServerOptions, InferenceServerConfig } from '@shared/src/models/InferenceServerConfig';
import { getFreeRandomPort } from './ports';
import { type InferenceServer, InferenceType } from '@shared/src/models/IInference';
import type { ModelInfo } from '@shared/src/models/IModelInfo';

export const LABEL_INFERENCE_SERVER: string = 'ai-lab-inference-server';

/**
 * Return container connection provider
 */
export function getProviderContainerConnection(providerId?: string): ProviderContainerConnection {
  // Get started providers
  const providers = provider
    .getContainerConnections()
    .filter(connection => connection.connection.status() === 'started');

  if (providers.length === 0) throw new Error('no engine started could be find.');

  let output: ProviderContainerConnection | undefined = undefined;

  // If we expect a specific engine
  if (providerId !== undefined) {
    output = providers.find(engine => engine.providerId === providerId);
  } else {
    // Have a preference for a podman engine
    output = providers.find(engine => engine.connection.type === 'podman');
    if (output === undefined) {
      output = providers[0];
    }
  }
  if (output === undefined) throw new Error('cannot find any started container provider.');
  return output;
}

/**
 * Given an image name, it will return the ImageInspectInfo corresponding. Will raise an error if not found.
 * @param connection
 * @param image
 * @param callback
 */
export async function getImageInfo(
  connection: ContainerProviderConnection,
  image: string,
  callback: (event: PullEvent) => void,
): Promise<ImageInfo> {
  let imageInfo: ImageInfo | undefined;
  try {
    // Pull image
    await containerEngine.pullImage(connection, image, callback);
    // Get image inspect
    imageInfo = (
      await containerEngine.listImages({
        provider: connection,
      } as ListImagesOptions)
    ).find(imageInfo => imageInfo.RepoTags?.some(tag => tag === image));
  } catch (err: unknown) {
    console.warn('Something went wrong while trying to get image inspect', err);
    throw err;
  }

  if (imageInfo === undefined) throw new Error(`image ${image} not found.`);

  return imageInfo;
}

export async function withDefaultConfiguration(
  options: CreationInferenceServerOptions,
): Promise<InferenceServerConfig> {
  if (options.modelsInfo.length === 0) throw new Error('modelsInfo need to contain at least one element.');

  return {
    port: options.port || (await getFreeRandomPort('0.0.0.0')),
    image: options.image,
    labels: options.labels || {},
    modelsInfo: options.modelsInfo,
    providerId: options.providerId,
    inferenceProvider: options.inferenceProvider,
    gpuLayers: options.gpuLayers,
  };
}

export function isTransitioning(server: InferenceServer): boolean {
  switch (server.status) {
    case 'deleting':
    case 'stopping':
    case 'starting':
      return true;
    default:
      break;
  }

  return false;
}

/**
 * Given a primitive (string) return the InferenceType enum
 * @param value
 */
export function parseInferenceType(value: string | undefined): InferenceType {
  if (!value) return InferenceType.NONE;
  return (Object.values(InferenceType) as unknown as string[]).includes(value)
    ? (value as unknown as InferenceType)
    : InferenceType.NONE;
}

/**
 * Let's collect the backend required by the provided models
 * we only support one backend for all the models, if multiple are provided, NONE will be return
 */
export function getInferenceType(modelsInfo: ModelInfo[]): InferenceType {
  const backends: InferenceType[] = modelsInfo.map(info => parseInferenceType(info.backend));
  if (new Set(backends).size !== 1) return InferenceType.NONE;

  return backends[0];
}
