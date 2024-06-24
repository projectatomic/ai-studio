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
import { vi, expect, test } from 'vitest';
import { configuration, type Configuration, type Webview } from '@podman-desktop/api';
import { ConfigurationRegistry } from './ConfigurationRegistry';

const fakeConfiguration = {
  get: vi.fn(),
  has: vi.fn(),
} as unknown as Configuration;

const webviewMock = {
  postMessage: vi.fn(),
} as unknown as Webview;

vi.mock('@podman-desktop/api', async () => {
  return {
    configuration: {
      getConfiguration: () => fakeConfiguration,
      onDidChangeConfiguration: vi.fn(),
    },
  };
});

test('init should init listener', () => {
  const registry = new ConfigurationRegistry(webviewMock, 'appdir');
  vi.mocked(fakeConfiguration.has).mockReturnValue(true);

  registry.init();
  expect(configuration.onDidChangeConfiguration).toHaveBeenCalled();
});

test('dispose should dispose listener', () => {
  const registry = new ConfigurationRegistry(webviewMock, 'appdir');
  vi.mocked(fakeConfiguration.has).mockReturnValue(true);

  const disposeMock = vi.fn();
  vi.mocked(configuration.onDidChangeConfiguration).mockReturnValue({ dispose: disposeMock });

  registry.init();
  expect(configuration.onDidChangeConfiguration).toHaveBeenCalled();

  registry.dispose();
  expect(disposeMock).toHaveBeenCalled();
});
