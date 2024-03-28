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
import { EventEmitter } from '@podman-desktop/api';

vi.mock('@podman-desktop/api', async () => {
  return {
    EventEmitter: vi.fn(),
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

test('token should call listener on cancel request', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());
  const listener = vi.fn();
  source.token.onCancellationRequested(listener);
  source.cancel();

  expect(listener).toHaveBeenCalledOnce();
});

test('disposing without cancel the token should not emit event', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());
  const listener = vi.fn();
  source.token.onCancellationRequested(listener);
  source.dispose(false);

  expect(listener).not.toHaveBeenCalled();
});

test('disposing with cancel the token should emit event', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());
  const listener = vi.fn();
  source.token.onCancellationRequested(listener);
  source.dispose(true);

  expect(listener).toHaveBeenCalled();
});

test('all listeners should be call on cancel', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());
  const listeners = [...Array(10).keys()].map(() => vi.fn());
  listeners.forEach(listener => {
    source.token.onCancellationRequested(listener);
  });
  source.cancel();

  listeners.forEach(listener => {
    expect(listener).toHaveBeenCalled();
  });
});

test('disposing registry should dispose with cancel all tokens', () => {
  const registry = new CancellationTokenRegistry();
  const source = registry.getCancellationTokenSource(registry.createCancellationTokenSource());
  const listener = vi.fn();
  source.token.onCancellationRequested(listener);

  registry.dispose();
  expect(listener).toHaveBeenCalled();
});
