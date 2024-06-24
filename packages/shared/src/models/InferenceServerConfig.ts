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
import type { ModelInfo } from './IModelInfo';

export type CreationInferenceServerOptions = Partial<InferenceServerConfig> & { modelsInfo: ModelInfo[] };

export interface InferenceServerConfig {
  /**
   * Port to expose
   */
  port: number;
  /**
   * The identifier of the container provider to use
   */
  providerId?: string;
  /**
   * The name of the inference provider to use
   */
  inferenceProvider?: string;
  /**
   * Image to use
   */
  image?: string;
  /**
   * Labels to use for the container
   */
  labels: { [id: string]: string };

  /**
   * Model info for the models
   */
  modelsInfo: ModelInfo[];
  /**
   * Number of layers to offload to the GPU
   * @default undefined the GPU will not be used
   * -1 to offload all the layers
   */
  gpuLayers?: number;
}
