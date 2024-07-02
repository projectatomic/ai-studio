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
import { expect, test, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import ModelColumnIcon from './ModelColumnIcon.svelte';
import { type InferenceServer, InferenceType } from '@shared/src/models/IInference';

const mocks = vi.hoisted(() => {
  return {
    getInferenceServersMock: vi.fn<void[], InferenceServer[]>(),
  };
});

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
});

test('Expect remote model to have NONE title', async () => {
  const object: ModelInfo = {
    id: 'model-downloaded-id',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    memory: 1000,
  };

  mocks.getInferenceServersMock.mockReturnValue([]);

  render(ModelColumnIcon, { object });

  const role = screen.getByRole('status');
  expect(role).toBeDefined();
  expect(role.title).toBe('NONE');
});

test('Expect downloaded model to have DOWNLOADED title', async () => {
  const object: ModelInfo = {
    id: 'model-downloaded-id',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: {
      file: 'file',
      creation: undefined,
      size: 1000,
      path: 'path',
    },
    memory: 1000,
  };

  mocks.getInferenceServersMock.mockReturnValue([]);

  render(ModelColumnIcon, { object });

  const role = screen.getByRole('status');
  expect(role).toBeDefined();
  expect(role.title).toBe('DOWNLOADED');
});

test('Expect in used model to have USED title', async () => {
  const object: ModelInfo = {
    id: 'model-in-used-id',
    description: '',
    hw: '',
    license: '',
    name: '',
    registry: '',
    url: '',
    file: {
      file: 'file',
      creation: undefined,
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
  render(ModelColumnIcon, { object });

  const role = screen.getByRole('status');
  expect(role).toBeDefined();
  expect(role.title).toBe('USED');
});
