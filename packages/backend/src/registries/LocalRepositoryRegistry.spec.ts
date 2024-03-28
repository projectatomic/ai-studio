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
import { LocalRepositoryRegistry } from './LocalRepositoryRegistry';
import { Messages } from '@shared/Messages';
import type { Webview } from '@podman-desktop/api';
import type { Recipe } from '@shared/src/models/IRecipe';
import fs from 'node:fs';
import path from 'node:path';

const mocks = vi.hoisted(() => ({
  DisposableCreateMock: vi.fn(),
  postMessageMock: vi.fn(),
}));

vi.mock('@podman-desktop/api', async () => {
  return {
    Disposable: {
      create: mocks.DisposableCreateMock,
    },
  };
});

vi.mock('node:fs', () => {
  return {
    existsSync: vi.fn(),
    promises: {
      rm: vi.fn(),
    },
  };
});

beforeEach(() => {
  vi.resetAllMocks();
  vi.mock('node:fs');
  mocks.postMessageMock.mockResolvedValue(undefined);
});

test('should not have any repositories by default', () => {
  const localRepositories = new LocalRepositoryRegistry(
    {
      postMessage: mocks.postMessageMock,
    } as unknown as Webview,
    '/appUserDirectory',
  );
  expect(localRepositories.getLocalRepositories().length).toBe(0);
});

test('should notify webview when register', () => {
  const localRepositories = new LocalRepositoryRegistry(
    {
      postMessage: mocks.postMessageMock,
    } as unknown as Webview,
    '/appUserDirectory',
  );
  localRepositories.register({ path: 'random', labels: { 'recipe-id': 'random' } });
  expect(mocks.postMessageMock).toHaveBeenNthCalledWith(1, {
    id: Messages.MSG_LOCAL_REPOSITORY_UPDATE,
    body: [{ path: 'random', labels: { 'recipe-id': 'random' } }],
  });
});

test('should notify webview when unregister', async () => {
  const localRepositories = new LocalRepositoryRegistry(
    {
      postMessage: mocks.postMessageMock,
    } as unknown as Webview,
    '/appUserDirectory',
  );
  vi.spyOn(fs.promises, 'rm').mockResolvedValue();
  localRepositories.register({ path: 'random', labels: { 'recipe-id': 'random' } });
  await localRepositories.deleteLocalRepository('random');

  expect(mocks.postMessageMock).toHaveBeenLastCalledWith({
    id: Messages.MSG_LOCAL_REPOSITORY_UPDATE,
    body: [],
  });
});

test('should register localRepo if it find the folder of the recipe', () => {
  vi.spyOn(fs, 'existsSync').mockReturnValue(true);
  const localRepositories = new LocalRepositoryRegistry(
    {
      postMessage: mocks.postMessageMock,
    } as unknown as Webview,
    '/appUserDirectory',
  );
  const registerMock = vi.spyOn(localRepositories, 'register');
  localRepositories.init([
    {
      id: 'recipe',
    } as unknown as Recipe,
  ]);

  expect(registerMock).toHaveBeenCalledWith({
    path: path.join('/appUserDirectory', 'recipe'),
    labels: { 'recipe-id': 'recipe' },
  });
});

test('should NOT register localRepo if it does not find the folder of the recipe', () => {
  vi.spyOn(fs, 'existsSync').mockReturnValue(false);
  const localRepositories = new LocalRepositoryRegistry(
    {
      postMessage: mocks.postMessageMock,
    } as unknown as Webview,
    '/appUserDirectory',
  );
  const registerMock = vi.spyOn(localRepositories, 'register');
  localRepositories.init([
    {
      id: 'recipe',
    } as unknown as Recipe,
  ]);

  expect(registerMock).not.toHaveBeenCalled();
});
