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
import ModelColumnActions from '/@/lib/table/model/ModelColumnActions.svelte';
import { router } from 'tinro';
import type { ModelInfoUI } from '/@/models/ModelInfoUI';

const mocks = vi.hoisted(() => ({
  requestRemoveLocalModel: vi.fn(),
  openFile: vi.fn(),
  downloadModel: vi.fn(),
}));

vi.mock('/@/utils/client', () => ({
  studioClient: {
    requestRemoveLocalModel: mocks.requestRemoveLocalModel,
    openFile: mocks.openFile,
    downloadModel: mocks.downloadModel,
  },
}));

beforeEach(() => {
  vi.resetAllMocks();

  mocks.downloadModel.mockResolvedValue(undefined);
  mocks.openFile.mockResolvedValue(undefined);
  mocks.requestRemoveLocalModel.mockResolvedValue(undefined);
});

test('Expect folder and delete button in document', async () => {
  const d = new Date();
  d.setDate(d.getDate() - 2);

  const object: ModelInfoUI = {
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
  const object: ModelInfoUI = {
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
  const object: ModelInfoUI = {
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

  const object: ModelInfoUI = {
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

test('Expect error tooltip to be shown if action failed', async () => {
  const object: ModelInfoUI = {
    id: 'my-model',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: undefined,
    memory: 1000,
    actionError: 'error while executing X',
  };
  render(ModelColumnActions, { object });

  const tooltip = screen.getByLabelText('tooltip');
  expect(tooltip).toBeInTheDocument();
  expect(tooltip.textContent).equals('error while executing X');
});
