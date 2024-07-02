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

import '@testing-library/jest-dom/vitest';
import { test, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import ModelColumnActions from '/@/lib/table/model/ModelColumnActions.svelte';
import { router } from 'tinro';
import { type InferenceServer, InferenceType } from '@shared/src/models/IInference';

const mocks = vi.hoisted(() => ({
  requestRemoveLocalModel: vi.fn(),
  openFile: vi.fn(),
  downloadModel: vi.fn(),
  getInferenceServersMock: vi.fn<void[], InferenceServer[]>(),
}));

vi.mock('/@/utils/client', () => ({
  studioClient: {
    requestRemoveLocalModel: mocks.requestRemoveLocalModel,
    openFile: mocks.openFile,
    downloadModel: mocks.downloadModel,
  },
}));

vi.mock('../../../stores/inferenceServers', () => ({
  inferenceServers: {
    subscribe: (f: (msg: InferenceServer[]) => void) => {
      f(mocks.getInferenceServersMock());
      return () => {};
    },
  },
}));

beforeEach(() => {
  vi.resetAllMocks();

  mocks.getInferenceServersMock.mockReturnValue([]);

  mocks.downloadModel.mockResolvedValue(undefined);
  mocks.openFile.mockResolvedValue(undefined);
  mocks.requestRemoveLocalModel.mockResolvedValue(undefined);
});

test('Expect folder and delete button in document', async () => {
  const d = new Date();
  d.setDate(d.getDate() - 2);

  const object: ModelInfo = {
    id: 'my-model',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: {
      file: 'file',
      creation: d,
      size: 1000,
      path: 'path',
    },
    memory: 1000,
  };
  render(ModelColumnActions, { object });

  const explorerBtn = screen.getByTitle('Open Model Folder');
  expect(explorerBtn).toBeInTheDocument();

  const deleteBtn = screen.getByTitle('Delete Model');
  expect(deleteBtn).toBeInTheDocument();

  const rocketBtn = screen.getByTitle('Create Model Service');
  expect(rocketBtn).toBeInTheDocument();

  const downloadBtn = screen.queryByTitle('Download Model');
  expect(downloadBtn).toBeNull();
});

test('Expect download button in document', async () => {
  const object: ModelInfo = {
    id: 'my-model',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: undefined,
    memory: 1000,
  };
  render(ModelColumnActions, { object });

  const explorerBtn = screen.queryByTitle('Open Model Folder');
  expect(explorerBtn).toBeNull();

  const deleteBtn = screen.queryByTitle('Delete Model');
  expect(deleteBtn).toBeNull();

  const rocketBtn = screen.queryByTitle('Create Model Service');
  expect(rocketBtn).toBeNull();

  const downloadBtn = screen.getByTitle('Download Model');
  expect(downloadBtn).toBeInTheDocument();
});

test('Expect downloadModel to be call on click', async () => {
  const object: ModelInfo = {
    id: 'my-model',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: undefined,
    memory: 1000,
  };
  render(ModelColumnActions, { object });

  const downloadBtn = screen.getByTitle('Download Model');
  expect(downloadBtn).toBeInTheDocument();

  await fireEvent.click(downloadBtn);
  await waitFor(() => {
    expect(mocks.downloadModel).toHaveBeenCalledWith('my-model');
  });
});

test('Expect router to be called when rocket icon clicked', async () => {
  const gotoMock = vi.spyOn(router, 'goto');
  const replaceMock = vi.spyOn(router.location.query, 'replace');

  const object: ModelInfo = {
    id: 'my-model',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: {
      file: 'file',
      creation: new Date(),
      size: 1000,
      path: 'path',
    },
    memory: 1000,
  };
  render(ModelColumnActions, { object });

  const rocketBtn = screen.getByTitle('Create Model Service');

  await fireEvent.click(rocketBtn);
  await waitFor(() => {
    expect(gotoMock).toHaveBeenCalledWith('/service/create');
    expect(replaceMock).toHaveBeenCalledWith({ 'model-id': 'my-model' });
  });
});

test('Expect delete button to be disabled when model in use', async () => {
  const object: ModelInfo = {
    id: 'my-model',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: {
      file: 'file',
      creation: new Date(),
      size: 1000,
      path: 'path',
    },
    memory: 1000,
  };

  mocks.getInferenceServersMock.mockReturnValue([
    {
      models: [object],
      type: InferenceType.LLAMA_CPP,
      status: 'running',
      container: {
        containerId: '',
        engineId: '',
      },
      connection: {
        port: 0,
      },
      health: undefined,
    },
  ]);
  render(ModelColumnActions, { object });

  const deleteBtn = screen.getByTitle('Delete Model');
  expect(deleteBtn).toBeDefined();

  await vi.waitFor(() => {
    // disable class
    expect(deleteBtn.classList).toContain('text-gray-900');
  });
});
