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

/* eslint-disable @typescript-eslint/no-explicit-any */

import { afterEach, beforeEach, expect, test, vi, describe } from 'vitest';
import { Studio } from './studio';
import { type ExtensionContext, EventEmitter, version } from '@podman-desktop/api';

import * as fs from 'node:fs';

vi.mock('./managers/modelsManager');

const mockedExtensionContext = {
  subscriptions: [],
  storagePath: 'dummy-storage-path',
} as unknown as ExtensionContext;

const studio = new Studio(mockedExtensionContext);

const mocks = vi.hoisted(() => ({
  listContainers: vi.fn(),
  getContainerConnections: vi.fn(),
  postMessage: vi.fn(),
  logErrorMock: vi.fn(),
  consoleWarnMock: vi.fn(),
  consoleLogMock: vi.fn(),
}));

vi.mock('../package.json', () => ({
  engines: {
    'podman-desktop': '>=1.0.0',
  },
}));

vi.mock('@podman-desktop/api', async () => {
  return {
    configuration: {
      getConfiguration: () => ({
        get: vi.fn(),
      }),
      onDidChangeConfiguration: vi.fn(),
    },
    version: '1.8.0',
    fs: {
      createFileSystemWatcher: () => ({
        onDidCreate: vi.fn(),
        onDidDelete: vi.fn(),
        onDidChange: vi.fn(),
      }),
    },
    EventEmitter: vi.fn(),
    Uri: class {
      static joinPath = () => ({ fsPath: '.' });
    },
    window: {
      createWebviewPanel: () => ({
        webview: {
          html: '',
          onDidReceiveMessage: vi.fn(),
          postMessage: mocks.postMessage,
        },
        onDidChangeViewState: vi.fn(),
      }),
    },
    env: {
      createTelemetryLogger: () => ({
        logUsage: vi.fn(),
        logError: mocks.logErrorMock,
      }),
    },
    containerEngine: {
      onEvent: vi.fn(),
      listContainers: mocks.listContainers,
    },
    provider: {
      onDidRegisterContainerConnection: vi.fn(),
      onDidUpdateContainerConnection: vi.fn(),
      getContainerConnections: mocks.getContainerConnections,
    },
    commands: {
      registerCommand: vi.fn(),
    },
    Disposable: {
      create: vi.fn(),
    },
  };
});

/// mock console.log
const originalConsoleLog = console.log;

beforeEach(() => {
  vi.clearAllMocks();
  console.log = mocks.consoleLogMock;
  console.warn = mocks.consoleWarnMock;

  vi.mocked(EventEmitter).mockReturnValue({
    event: vi.fn(),
    fire: vi.fn(),
  } as unknown as EventEmitter<unknown>);

  mocks.postMessage.mockResolvedValue(undefined);
});

afterEach(() => {
  console.log = originalConsoleLog;
});

test('check activate', async () => {
  expect(version).toBe('1.8.0');
  mocks.listContainers.mockReturnValue([]);
  mocks.getContainerConnections.mockReturnValue([]);
  vi.spyOn(fs.promises, 'readFile').mockImplementation(() => {
    return Promise.resolve('<html></html>');
  });
  await studio.activate();

  // expect the activate method to be called on the studio class
  expect(mocks.consoleLogMock).toBeCalledWith('starting AI Lab extension');
});

describe('version checker', () => {
  test('check activate incompatible', async () => {
    (version as string) = '0.7.0';
    await expect(studio.activate()).rejects.toThrowError(
      'Extension is not compatible with Podman Desktop version below 1.0.0. Current 0.7.0',
    );

    // expect the activate method to be called on the studio class
    expect(mocks.logErrorMock).toBeCalledWith('start.incompatible', {
      version: '0.7.0',
      message: 'error activating extension on version below 1.0.0',
    });
  });

  test('version undefined', async () => {
    (version as string | undefined) = undefined;
    await expect(studio.activate()).rejects.toThrowError(
      'Extension is not compatible with Podman Desktop version below 1.0.0. Current unknown',
    );

    // expect the activate method to be called on the studio class
    expect(mocks.logErrorMock).toBeCalledWith('start.incompatible', {
      version: 'unknown',
      message: 'error activating extension on version below 1.0.0',
    });
  });

  test('check activate next value', async () => {
    (version as string) = '1.0.1-next';
    await studio.activate();

    expect(mocks.logErrorMock).not.toHaveBeenCalled();
  });

  /**
   * This check ensure we do not support old nighties version to be used
   * update introduced in https://github.com/containers/podman-desktop/pull/7643
   */
  test('check activate old nighties value', async () => {
    (version as string) = 'v0.0.202404030805-3cb4544';
    await expect(studio.activate()).rejects.toThrowError(
      'Extension is not compatible with Podman Desktop version below 1.0.0. Current v0.0.202404030805-3cb4544',
    );

    expect(mocks.logErrorMock).toHaveBeenCalled();
  });

  test('check activate version nighties', async () => {
    (version as string) = `1.0.0-${Date.now()}-b35e7bef`;
    await studio.activate();

    expect(mocks.logErrorMock).not.toHaveBeenCalled();
  });
});

test('check deactivate ', async () => {
  await studio.deactivate();

  // expect the deactivate method to be called on the studio class
  expect(mocks.consoleLogMock).toBeCalledWith('stopping AI Lab extension');
});
