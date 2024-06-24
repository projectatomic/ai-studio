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
import { configuration, type Configuration, type Disposable, type Webview } from '@podman-desktop/api';
import { Publisher } from '../utils/Publisher';
import type { ExtensionConfiguration } from '@shared/src/models/IExtensionConfiguration';
import { Messages } from '@shared/Messages';
import path from 'node:path';

const CONFIGURATION_SECTIONS: string[] = ['ai-lab.experimentalGPU', 'ai-lab.models.path'];

export class ConfigurationRegistry extends Publisher<ExtensionConfiguration> implements Disposable {
  #configuration: Configuration;
  #configurationDisposable: Disposable | undefined;

  constructor(
    webview: Webview,
    private appUserDirectory: string,
  ) {
    super(webview, Messages.MSG_CONFIGURATION_UPDATE, () => this.getExtensionConfiguration());

    this.#configuration = configuration.getConfiguration('ai-lab');
  }

  getExtensionConfiguration(): ExtensionConfiguration {
    return {
      experimentalGPU: this.#configuration.get<boolean>('experimentalGPU') ?? false,
      modelsPath: this.getModelsPath(),
    };
  }

  private getModelsPath(): string {
    const value = this.#configuration.get<string>('models.path');
    if (value && value.length > 0) {
      return value;
    }
    return path.join(this.appUserDirectory, 'models');
  }

  dispose(): void {
    this.#configurationDisposable?.dispose();
  }

  init(): void {
    this.#configurationDisposable = configuration.onDidChangeConfiguration(event => {
      if (CONFIGURATION_SECTIONS.some(section => event.affectsConfiguration(section))) {
        this.notify();
      }
    });
  }
}
