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
import { vi, test, expect, beforeEach } from 'vitest';
import { screen, render } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import type { ApplicationCatalog } from '@shared/src/models/IApplicationCatalog';
import * as catalogStore from '/@/stores/catalog';
import { readable, writable } from 'svelte/store';
import RecipeDetails from './RecipeDetails.svelte';
import * as tasksStore from '../stores/tasks';
import type { Task } from '@shared/src/models/ITask';

const mocks = vi.hoisted(() => {
  return {
    getLocalRepositoriesMock: vi.fn(),
    getTasksMock: vi.fn(),
    openFileMock: vi.fn(),
    requestDeleteLocalRepositoryMock: vi.fn(),
  };
});

vi.mock('../utils/client', async () => {
  return {
    studioClient: {
      openFile: mocks.openFileMock,
      requestDeleteLocalRepository: mocks.requestDeleteLocalRepositoryMock,
    },
    rpcBrowser: {
      subscribe: () => {
        return {
          unsubscribe: () => {},
        };
      },
    },
  };
});

vi.mock('/@/stores/tasks', async () => {
  return {
    tasks: vi.fn(),
  };
});

vi.mock('/@/stores/catalog', async () => {
  return {
    catalog: vi.fn(),
  };
});

vi.mock('../stores/localRepositories', () => ({
  localRepositories: {
    subscribe: (f: (msg: any) => void) => {
      f(mocks.getLocalRepositoriesMock());
      return () => {};
    },
  },
}));

const initialCatalog: ApplicationCatalog = {
  categories: [],
  models: [],
  recipes: [
    {
      id: 'recipe 1',
      name: 'Recipe 1',
      readme: 'readme 1',
      categories: [],
      recommended: ['model1', 'model2'],
      description: 'description 1',
      repository: 'repo 1',
    },
    {
      id: 'recipe 2',
      name: 'Recipe 2',
      readme: 'readme 2',
      categories: [],
      description: 'description 2',
      repository: 'repo 2',
    },
  ],
};

beforeEach(() => {
  vi.resetAllMocks();
  mocks.getLocalRepositoriesMock.mockReturnValue([]);
  const tasksList = writable<Task[]>([]);
  vi.mocked(tasksStore).tasks = tasksList;
});

test('button vs code should be visible if local repository is not empty', async () => {
  mocks.getLocalRepositoriesMock.mockReturnValue([
    {
      path: 'random-path',
      labels: {
        'recipe-id': 'recipe 1',
      },
    },
  ]);
  vi.mocked(catalogStore).catalog = readable<ApplicationCatalog>(initialCatalog);
  render(RecipeDetails, {
    recipeId: 'recipe 1',
  });

  const button = screen.getByTitle('Open in VS Code Desktop');
  expect(button).toBeDefined();
});

test('local clone and delete local clone buttons should be visible if local repository is not empty', async () => {
  mocks.getLocalRepositoriesMock.mockReturnValue([
    {
      path: 'random-path',
      labels: {
        'recipe-id': 'recipe 1',
      },
    },
  ]);
  vi.mocked(catalogStore).catalog = readable<ApplicationCatalog>(initialCatalog);
  render(RecipeDetails, {
    recipeId: 'recipe 1',
  });

  const buttonLocalClone = screen.getByRole('button', { name: 'Local clone' });
  expect(buttonLocalClone).toBeDefined();
  expect(buttonLocalClone).toBeInTheDocument();
  await userEvent.click(buttonLocalClone);

  expect(mocks.openFileMock).toBeCalled();

  const buttonDeleteClone = screen.getByTitle('Delete local clone');
  expect(buttonDeleteClone).toBeDefined();
  expect(buttonDeleteClone).toBeInTheDocument();
  await userEvent.click(buttonDeleteClone);

  expect(mocks.requestDeleteLocalRepositoryMock).toBeCalled();
});
