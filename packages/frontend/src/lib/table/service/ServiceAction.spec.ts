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

import { expect, test, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import ServiceAction from './ServiceAction.svelte';
import { studioClient } from '/@/utils/client';
import { InferenceType } from '@shared/src/models/IInference';

vi.mock('../../../utils/client', async () => ({
  studioClient: {
    startInferenceServer: vi.fn(),
    stopInferenceServer: vi.fn(),
    requestDeleteInferenceServer: vi.fn(),
  },
}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(studioClient.startInferenceServer).mockResolvedValue(undefined);
  vi.mocked(studioClient.stopInferenceServer).mockResolvedValue(undefined);
  vi.mocked(studioClient.requestDeleteInferenceServer).mockResolvedValue(undefined);
});

test('should display stop button when status running', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'running',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
  });

  const stopBtn = screen.getByTitle('Stop service');
  expect(stopBtn).toBeDefined();
});

test('should display start button when status stopped', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'stopped',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
  });

  const startBtn = screen.getByTitle('Start service');
  expect(startBtn).toBeDefined();
});

test('should call stopInferenceServer when click stop', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'running',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
  });

  const stopBtn = screen.getByTitle('Stop service');
  await fireEvent.click(stopBtn);
  expect(studioClient.stopInferenceServer).toHaveBeenCalledWith('dummyContainerId');
});

test('should call startInferenceServer when click start', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'stopped',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
  });

  const startBtn = screen.getByTitle('Start service');
  await fireEvent.click(startBtn);
  expect(studioClient.startInferenceServer).toHaveBeenCalledWith('dummyContainerId');
});

test('should call deleteInferenceServer when click delete', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'stopped',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
  });

  const startBtn = screen.getByTitle('Delete service');
  await fireEvent.click(startBtn);
  expect(studioClient.requestDeleteInferenceServer).toHaveBeenCalledWith('dummyContainerId');
});

test('should be disabled on transition', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'stopping',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
  });

  const startBtn = screen.getByTitle('Start service');
  expect(startBtn.classList).toContain('text-[var(--pd-action-button-disabled-text)]');

  const deleteBtn = screen.getByTitle('Delete service');
  expect(deleteBtn.classList).toContain('text-[var(--pd-action-button-disabled-text)]');
});

test('should have background on details', async () => {
  render(ServiceAction, {
    object: {
      type: InferenceType.LLAMA_CPP,
      health: undefined,
      models: [],
      connection: { port: 8888 },
      status: 'stopped',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
    },
    detailed: true,
  });

  const startBtn = screen.getByTitle('Start service');
  expect(startBtn.classList).toContain('bg-[var(--pd-action-button-details-bg)]');
  expect(startBtn.classList).toContain('rounded-lg');
});
