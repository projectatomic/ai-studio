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
import type { ContainerCreateOptions, DeviceRequest, ImageInfo, MountConfig } from '@podman-desktop/api';
import type { InferenceServerConfig } from '@shared/src/models/InferenceServerConfig';
import { type BetterContainerCreateResult, InferenceProvider } from './InferenceProvider';
import { getModelPropertiesForEnvironment } from '../../utils/modelsUtils';
import { DISABLE_SELINUX_LABEL_SECURITY_OPTION } from '../../utils/utils';
import { LABEL_INFERENCE_SERVER } from '../../utils/inferenceUtils';
import type { TaskRegistry } from '../../registries/TaskRegistry';
import { InferenceType } from '@shared/src/models/IInference';
import type { GPUManager } from '../../managers/GPUManager';
import type { IGPUInfo } from '@shared/src/models/IGPUInfo';

export const LLAMA_CPP_CPU = 'ghcr.io/containers/llamacpp_python:latest';
export const LLAMA_CPP_CUDA = 'ghcr.io/containers/llamacpp_python_cuda:latest';

export const SECOND: number = 1_000_000_000;

interface Device {
  PathOnHost: string;
  PathInContainer: string;
  CgroupPermissions: string;
}

export class LlamaCppPython extends InferenceProvider {
  constructor(
    taskRegistry: TaskRegistry,
    private gpuManager: GPUManager,
  ) {
    super(taskRegistry, InferenceType.LLAMA_CPP, 'LLama-cpp');
  }

  dispose() {}

  public enabled = (): boolean => true;

  protected async getContainerCreateOptions(
    config: InferenceServerConfig,
    imageInfo: ImageInfo,
    gpu?: IGPUInfo,
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

    const mounts: MountConfig = [
      {
        Target: '/models',
        Source: modelInfo.file.path,
        Type: 'bind',
      },
    ];

    const deviceRequests: DeviceRequest[] = [];
    const devices: Device[] = [];
    let entrypoint: string | undefined = undefined;
    let cmd: string[] = [];
    let user: string | undefined = undefined;

    // specific to WSL
    if (gpu) {
      mounts.push({
        Target: '/usr/lib/wsl',
        Source: '/usr/lib/wsl',
        Type: 'bind',
      });

      // adding gpu capabilities
      deviceRequests.push({
        Capabilities: [['gpu']],
        Count: -1, // -1: all
      });

      devices.push({
        PathOnHost: '/dev/dxg',
        PathInContainer: '/dev/dxg',
        CgroupPermissions: 'r',
      });

      entrypoint = '/usr/bin/sh';
      envs.push(`GPU_LAYERS=${config.gpuLayers}`);

      user = '0';

      cmd = [
        '-c',
        '/usr/bin/ln -s /usr/lib/wsl/lib/* /usr/lib64/ && PATH="${PATH}:/usr/lib/wsl/lib/" && chmod 755 ./run.sh && ./run.sh',
      ];
    }

    return {
      Image: imageInfo.Id,
      Detach: true,
      Entrypoint: entrypoint,
      User: user,
      ExposedPorts: { [`${config.port}`]: {} },
      HostConfig: {
        AutoRemove: false,
        Devices: devices,
        Mounts: mounts,
        DeviceRequests: deviceRequests,
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
      Cmd: cmd,
    };
  }

  async perform(config: InferenceServerConfig): Promise<BetterContainerCreateResult> {
    if (!this.enabled()) throw new Error('not enabled');

    let gpu: IGPUInfo | undefined = undefined;

    // get the first GPU if requested
    if ((config.gpuLayers ?? 0) !== 0) {
      const gpus: IGPUInfo[] = await this.gpuManager.collectGPUs();
      if (gpus.length === 0) throw new Error('no gpu was found.');
      if (gpus.length > 1)
        console.warn(`found ${gpus.length} gpus: using multiple GPUs is not supported. Using ${gpus[0].model}.`);
      gpu = gpus[0];
    }

    // pull the image
    const imageInfo: ImageInfo = await this.pullImage(
      config.providerId,
      config.image ?? (gpu ? LLAMA_CPP_CUDA : LLAMA_CPP_CPU),
      config.labels,
    );

    // Get the container creation options
    const containerCreateOptions: ContainerCreateOptions = await this.getContainerCreateOptions(config, imageInfo, gpu);

    // Create the container
    return this.createContainer(imageInfo.engineId, containerCreateOptions, config.labels);
  }
}
