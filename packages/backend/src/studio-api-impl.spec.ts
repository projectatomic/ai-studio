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

import { beforeEach, expect, test, vi } from 'vitest';
import content from './tests/ai-test.json';
import type { ApplicationManager } from './managers/applicationManager';
import { StudioApiImpl } from './studio-api-impl';
import type { InferenceManager } from './managers/inference/inferenceManager';
import type { TelemetryLogger, Webview } from '@podman-desktop/api';
import { window, EventEmitter } from '@podman-desktop/api';
import { CatalogManager } from './managers/catalogManager';
import type { ModelsManager } from './managers/modelsManager';
import { timeout } from './utils/utils';
import type { TaskRegistry } from './registries/TaskRegistry';
import { LocalRepositoryRegistry } from './registries/LocalRepositoryRegistry';
import type { Recipe } from '@shared/src/models/IRecipe';
import type { PlaygroundV2Manager } from './managers/playgroundV2Manager';
import type { SnippetManager } from './managers/SnippetManager';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import type { CancellationTokenRegistry } from './registries/CancellationTokenRegistry';

vi.mock('./ai.json', () => {
  return {
    default: content,
  };
});

vi.mock('node:fs', () => {
  return {
    existsSync: vi.fn(),
    promises: {
      readFile: vi.fn(),
    },
  };
});

const mocks = vi.hoisted(() => ({
  withProgressMock: vi.fn(),
  showWarningMessageMock: vi.fn(),
  deleteApplicationMock: vi.fn(),
}));

vi.mock('@podman-desktop/api', async () => {
  return {
    EventEmitter: vi.fn(),
    window: {
      withProgress: mocks.withProgressMock,
      showWarningMessage: mocks.showWarningMessageMock,
      showErrorMessage: vi.fn(),
    },
    ProgressLocation: {
      TASK_WIDGET: 'TASK_WIDGET',
    },
    fs: {
      createFileSystemWatcher: () => ({
        onDidCreate: vi.fn(),
        onDidDelete: vi.fn(),
        onDidChange: vi.fn(),
      }),
    },
  };
});

let studioApiImpl: StudioApiImpl;
let catalogManager: CatalogManager;
let localRepositoryRegistry: LocalRepositoryRegistry;

beforeEach(async () => {
  vi.resetAllMocks();

  const appUserDirectory = '.';

  // Creating CatalogManager
  catalogManager = new CatalogManager(
    {
      postMessage: vi.fn().mockResolvedValue(undefined),
    } as unknown as Webview,
    appUserDirectory,
  );

  localRepositoryRegistry = new LocalRepositoryRegistry(
    {
      postMessage: vi.fn().mockResolvedValue(undefined),
    } as unknown as Webview,
    appUserDirectory,
  );

  // Creating StudioApiImpl
  studioApiImpl = new StudioApiImpl(
    {
      deleteApplication: mocks.deleteApplicationMock,
    } as unknown as ApplicationManager,
    catalogManager,
    {} as unknown as ModelsManager,
    {} as TelemetryLogger,
    localRepositoryRegistry,
    {} as unknown as TaskRegistry,
    {} as unknown as InferenceManager,
    {} as unknown as PlaygroundV2Manager,
    {} as unknown as SnippetManager,
    {} as unknown as CancellationTokenRegistry,
  );
  vi.mock('node:fs');

  const listeners: ((value: unknown) => void)[] = [];

  vi.mocked(EventEmitter).mockReturnValue({
    event: vi.fn().mockImplementation(callback => {
      listeners.push(callback);
    }),
    fire: vi.fn().mockImplementation((content: unknown) => {
      listeners.forEach(listener => listener(content));
    }),
  } as unknown as EventEmitter<unknown>);
});

test('expect pull application to call the withProgress api method', async () => {
  vi.spyOn(catalogManager, 'getRecipes').mockReturnValue([
    {
      id: 'recipe 1',
    } as unknown as Recipe,
  ]);
  vi.spyOn(catalogManager, 'getModelById').mockReturnValue({
    id: 'model',
  } as unknown as ModelInfo);

  mocks.withProgressMock.mockResolvedValue(undefined);

  await studioApiImpl.pullApplication('recipe 1', 'model1');
  expect(mocks.withProgressMock).toHaveBeenCalledOnce();
});

test('requestRemoveApplication should ask confirmation', async () => {
  vi.spyOn(catalogManager, 'getRecipeById').mockReturnValue({
    name: 'Recipe 1',
  } as unknown as Recipe);
  mocks.showWarningMessageMock.mockResolvedValue('Confirm');
  await studioApiImpl.requestRemoveApplication('recipe-id-1', 'model-id-1');
  await timeout(0);
  expect(mocks.deleteApplicationMock).toHaveBeenCalled();
});

test('requestDeleteLocalRepository should ask confirmation', async () => {
  mocks.showWarningMessageMock.mockResolvedValue('Confirm');
  const deleteLocalRepositoryMock = vi.spyOn(localRepositoryRegistry, 'deleteLocalRepository').mockResolvedValue();
  await studioApiImpl.requestDeleteLocalRepository('path');
  await timeout(0);
  expect(deleteLocalRepositoryMock).toHaveBeenCalled();
});

test('if requestDeleteLocalRepository fails an errorMessage should show up', async () => {
  mocks.showWarningMessageMock.mockResolvedValue('Confirm');
  const deleteLocalRepositoryMock = vi
    .spyOn(localRepositoryRegistry, 'deleteLocalRepository')
    .mockRejectedValue('error deleting');
  const errorMessageMock = vi.spyOn(window, 'showErrorMessage').mockResolvedValue('');
  await studioApiImpl.requestDeleteLocalRepository('path');
  await timeout(0);
  expect(deleteLocalRepositoryMock).toHaveBeenCalled();
  expect(errorMessageMock).toBeCalledWith('Error deleting local path "path". Error: error deleting');
});
