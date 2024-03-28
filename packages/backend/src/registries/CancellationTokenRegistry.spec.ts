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
import { beforeEach, expect, test, vi } from 'vitest';
import { CancellationTokenRegistry } from './CancellationTokenRegistry';
import { CancellationTokenSource, EventEmitter } from '@podman-desktop/api';

vi.mock('@podman-desktop/api', async () => {
  return {
    EventEmitter: vi.fn(),
    CancellationTokenSource: vi.fn(),
  };
});

beforeEach(() => {
  vi.resetAllMocks();

  // mock event emitters
  const listeners: ((value: unknown) => void)[] = [];

  vi.mocked(EventEmitter).mockReturnValue({
    event: vi.fn().mockImplementation(callback => {
      listeners.push(callback);
    }),
    dispose: vi.fn(),
    fire: vi.fn().mockImplementation((content: unknown) => {
      listeners.forEach(listener => listener(content));
    }),
  } as unknown as EventEmitter<unknown>);

  vi.mocked(CancellationTokenSource).mockReturnValue({
    cancel: vi.fn(),
    dispose: vi.fn(),
    token: {
      isCancellationRequested: false,
      onCancellationRequested: vi.fn(),
    },
  });
});

test('created token should be retrievable', () => {
  const registry = new CancellationTokenRegistry();
  const tokenId = registry.createCancellationTokenSource();
  expect(tokenId).toBeDefined();
  expect(registry.hasCancellationTokenSource(tokenId)).toBeTruthy();
});

test('created token should not be cancelled', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());
  expect(source).toBeDefined();
  expect(source.token.isCancellationRequested).toBeFalsy();
});

test('cancel token should be removed from registry', () => {
  const registry = new CancellationTokenRegistry();
  const tokenId = registry.createCancellationTokenSource();

  expect(registry.hasCancellationTokenSource(tokenId)).toBeTruthy();

  registry.cancel(tokenId);

  expect(registry.hasCancellationTokenSource(tokenId)).toBeFalsy();
});

test('disposing registry should dispose with cancel all tokens', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());

  registry.dispose();
  expect(source.cancel).toHaveBeenCalled();
  expect(source.dispose).toHaveBeenCalled();
});
