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

import { expect, test, vi, describe } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import ServiceStatus from './ServiceStatus.svelte';
import { studioClient } from '/@/utils/client';
import { type InferenceServerStatus, InferenceType } from '@shared/src/models/IInference';

vi.mock('../../../utils/client', async () => ({
  studioClient: {
    navigateToContainer: vi.fn(),
  },
}));

describe('transition statuses', () => {
  test.each(['starting', 'stopping', 'deleting'] as InferenceServerStatus[])(
    'status %s should display a spinner',
    status => {
      render(ServiceStatus, {
        object: {
          health: undefined,
          models: [],
          connection: { port: 8888 },
          status: status,
          container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
          type: InferenceType.LLAMA_CPP,
        },
      });

      const img = screen.getByRole('img');
      expect(img).toBeDefined();

      const button = screen.queryByRole('button');
      expect(button).toBeNull();
    },
  );
});

describe('stable statuses', () => {
  test.each(['running', 'stopped', 'error'] as InferenceServerStatus[])(
    'status %s should not display a spinner',
    status => {
      render(ServiceStatus, {
        object: {
          health: undefined,
          models: [],
          connection: { port: 8888 },
          status: status,
          container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
          type: InferenceType.LLAMA_CPP,
        },
      });

      const img = screen.queryByRole('img');
      expect(img).toBeNull();

      const button = screen.getByRole('button');
      expect(button).toBeDefined();
    },
  );
});

test('defined health should not display a spinner', async () => {
  render(ServiceStatus, {
    object: {
      health: {
        Status: 'starting',
        Log: [],
        FailingStreak: 1,
      },
      models: [],
      connection: { port: 8888 },
      status: 'running',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
      type: InferenceType.LLAMA_CPP,
    },
  });

  const img = screen.queryByRole('img');
  expect(img).toBeNull();

  const button = screen.getByRole('button');
  expect(button).toBeDefined();
});

test('click on status icon should redirect to container', async () => {
  render(ServiceStatus, {
    object: {
      health: {
        Status: 'starting',
        Log: [],
        FailingStreak: 1,
      },
      models: [],
      connection: { port: 8888 },
      status: 'running',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
      type: InferenceType.LLAMA_CPP,
    },
  });
  // Get button and click on it
  const button = screen.getByRole('button');
  await fireEvent.click(button);

  await waitFor(() => {
    expect(studioClient.navigateToContainer).toHaveBeenCalledWith('dummyContainerId');
  });
});

test('error status should show degraded', async () => {
  render(ServiceStatus, {
    object: {
      models: [],
      connection: { port: 8888 },
      status: 'error',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
      type: InferenceType.LLAMA_CPP,
    },
  });
  // Get button and click on it
  const status = screen.getByRole('status');
  expect(status.title).toBe('DEGRADED');
});

test('running status with no healthcheck should show starting', async () => {
  render(ServiceStatus, {
    object: {
      models: [],
      connection: { port: 8888 },
      status: 'running',
      container: { containerId: 'dummyContainerId', engineId: 'dummyEngineId' },
      type: InferenceType.LLAMA_CPP,
    },
  });
  // Get button and click on it
  const status = screen.getByRole('status');
  expect(status.title).toBe('STARTING');
});
