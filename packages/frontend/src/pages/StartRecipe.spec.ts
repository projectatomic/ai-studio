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
import { vi, beforeEach, test, expect } from 'vitest';
import { studioClient } from '/@/utils/client';
import { render, screen, fireEvent } from '@testing-library/svelte';
import StartRecipe from '/@/pages/StartRecipe.svelte';
import type { Recipe } from '@shared/src/models/IRecipe';
import { InferenceType } from '@shared/src/models/IInference';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import type { Task } from '@shared/src/models/ITask';
import { router } from 'tinro';

const mocks = vi.hoisted(() => {
  return {
    // models store
    getModelsInfoMock: vi.fn(),
    // tasks store
    getTasksMock: vi.fn(),
    // local repository mock
    getLocalRepositoriesMock: vi.fn(),
    // catalog store
    getCatalogMock: vi.fn(),
  };
});

// Mock LocalRepository store
vi.mock('../stores/localRepositories', () => ({
  localRepositories: {
    subscribe: (f: (msg: any) => void) => {
      f(mocks.getLocalRepositoriesMock());
      return () => {};
    },
  },
}));

// Mock ModelsInfo store
vi.mock('../stores/modelsInfo', async () => {
  return {
    modelsInfo: {
      subscribe: (f: (msg: any) => void) => {
        f(mocks.getModelsInfoMock());
        return () => {};
      },
    },
  };
});

vi.mock('../stores/tasks', async () => {
  return {
    tasks: {
      subscribe: (f: (msg: any) => void) => {
        f(mocks.getTasksMock());
        return () => {};
      },
    },
  };
});

vi.mock('/@/stores/catalog', async () => {
  return {
    catalog: {
      subscribe: (f: (msg: any) => void) => {
        f(mocks.getCatalogMock());
        return () => {};
      },
    },
  };
});

vi.mock('tinro', () => ({
  router: {
    goto: vi.fn(),
    location: {
      query: new Map(),
    },
  },
}));

vi.mock('../utils/client', async () => ({
  studioClient: {
    checkContainerConnectionStatusAndResources: vi.fn(),
    requestPullApplication: vi.fn(),
  },
}));

const fakeRecipe: Recipe = {
  id: 'dummy-recipe-id',
  backend: InferenceType.LLAMA_CPP,
  name: 'Dummy Recipe',
  description: 'Dummy description',
  recommended: ['dummy-model-1'],
  categories: [],
} as unknown as Recipe;

const fakeRecommendedModel: ModelInfo = {
  id: 'dummy-model-1',
  backend: InferenceType.LLAMA_CPP,
  name: 'Dummy Model 1',
  file: {
    file: 'dummy-model-file',
    path: 'dummy-model-path',
  },
} as unknown as ModelInfo;

const fakeRemoteModel: ModelInfo = {
  id: 'dummy-model-2',
  backend: InferenceType.LLAMA_CPP,
  name: 'Dummy Model 2',
} as unknown as ModelInfo;

beforeEach(() => {
  vi.resetAllMocks();

  // reset all query between tests
  router.location.query.clear();

  mocks.getCatalogMock.mockReturnValue({
    recipes: [fakeRecipe],
  });

  mocks.getModelsInfoMock.mockReturnValue([fakeRecommendedModel, fakeRemoteModel]);

  // no local repository
  mocks.getLocalRepositoriesMock.mockReturnValue([]);

  mocks.getTasksMock.mockReturnValue([]);

  vi.mocked(studioClient.requestPullApplication).mockResolvedValue('fake-tracking-id');

  // mock scrollIntoView
  window.HTMLElement.prototype.scrollIntoView = function () {};
});

test('Recipe name should be visible', async () => {
  render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  const span = screen.getByLabelText('Recipe name');
  expect(span).toBeDefined();
  expect(span.textContent).toBe(fakeRecipe.name);
});

test('Recipe Local Repository should be visible when defined', async () => {
  // mock recipe local repository
  mocks.getLocalRepositoriesMock.mockReturnValue([
    {
      path: 'dummy-recipe-path',
      sourcePath: 'dummy-recipe-path',
      labels: {
        'recipe-id': fakeRecipe.id,
      },
    },
  ]);

  render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  const span = screen.getByLabelText('Recipe local path');
  expect(span).toBeDefined();
  expect(span.textContent).toBe('dummy-recipe-path');
});

test('Submit button should be disabled when no model is selected', async () => {
  mocks.getModelsInfoMock.mockReturnValue([]);

  render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  const button = screen.getByTitle(`Start ${fakeRecipe.name} recipe`);
  expect(button).toBeDefined();
  expect(button).toBeDisabled();
});

test('First recommended model should be selected as default model', async () => {
  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  await vi.waitFor(() => {
    const option = getSelectedOption<{ value: string }>(container);
    expect(option?.value).toBe(fakeRecommendedModel.id);
  });
});

/**
 * Return the selected value
 * @param container
 */
function getSelectedOption<T>(container: HTMLElement): T | undefined {
  const input = container.querySelector('input[name="select-model"][type="hidden"]');
  if (!input) throw new Error('input not found');
  if ((input as HTMLInputElement).value === undefined) return undefined;
  return JSON.parse((input as HTMLInputElement).value);
}

/**
 * Utility method to select an option in the svelte-select component
 * @param container
 * @param label
 */
async function selectOption(container: HTMLElement, label: string): Promise<void> {
  // first get the select input
  const input = screen.getByLabelText('Select Model');
  await fireEvent.pointerUp(input); // they are using the pointer up event instead of click.

  // get all options available
  const items = container.querySelectorAll('div[class~="list-item"]');
  // ensure we have two options
  expect(items.length).toBeGreaterThan(0);

  // get the option we are interested in (remote model here)
  const remoteModelOption = Array.from(items).find(item => item.querySelector('span')?.textContent === label);
  if (!remoteModelOption) throw new Error('missing options in select');

  // click on it
  await fireEvent.click(remoteModelOption);

  return await vi.waitFor(() => {
    const value = getSelectedOption<{ label: string }>(container);
    expect(value?.label).toBe(label);
  });
}

test('Selecting model not downloaded should display a warning', async () => {
  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  await selectOption(container, fakeRemoteModel.name);

  // Ensure the selected value is the one we choose
  await vi.waitFor(() => {
    const span = screen.getByRole('alert');
    expect(span).toBeDefined();
    expect(span.textContent).toBe(
      'The selected model will be downloaded. This action can take some time depending on your connection',
    );
  });
});

test('Selecting model downloaded should not display a warning', async () => {
  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  await selectOption(container, fakeRecommendedModel.name);
  const span = screen.queryByRole('alert');
  expect(span).toBeNull();
});

test('Selecting model should enable submit button', async () => {
  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  await selectOption(container, fakeRecommendedModel.name);

  const button = screen.getByTitle(`Start ${fakeRecipe.name} recipe`);
  expect(button).toBeDefined();
  expect(button).not.toBeDisabled();
});

test('Submit button should call requestPullApplication with proper arguments', async () => {
  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  await selectOption(container, fakeRecommendedModel.name);

  const button = screen.getByTitle(`Start ${fakeRecipe.name} recipe`);
  await fireEvent.click(button);

  await vi.waitFor(() => {
    expect(studioClient.requestPullApplication).toHaveBeenCalledWith(fakeRecipe.id, fakeRecommendedModel.id);
  });
});

test('Loading task should make the submit button disabled', async () => {
  mocks.getTasksMock.mockReturnValue([
    {
      id: 'dummy-task-id',
      name: 'Dummy task',
      state: 'loading',
      labels: {
        trackingId: 'fake-tracking-id',
      },
    } as Task,
  ]);
  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  await selectOption(container, fakeRecommendedModel.name);

  const button = screen.getByTitle(`Start ${fakeRecipe.name} recipe`);
  expect(button).not.toBeDisabled();
  await fireEvent.click(button);

  await vi.waitFor(() => {
    expect(button).toBeDefined();
  });
});

test('trackingId in router query should use it to display related tasks', () => {
  mocks.getTasksMock.mockReturnValue([
    {
      id: 'dummy-task-id',
      name: 'Dummy task',
      state: 'loading',
      labels: {
        trackingId: 'fake-tracking-id',
      },
    } as Task,
  ]);

  router.location.query.set('trackingId', 'fake-tracking-id');

  render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });
  const button = screen.getByTitle(`Start ${fakeRecipe.name} recipe`);
  expect(button).toBeDisabled();
});

test('restoring page should use model-id from tasks to restore the value in the select input', async () => {
  mocks.getTasksMock.mockReturnValue([
    {
      id: 'dummy-task-id',
      name: 'Dummy task',
      state: 'loading',
      labels: {
        trackingId: 'fake-tracking-id',
        'model-id': fakeRecommendedModel.id,
      },
    } as Task,
  ]);

  router.location.query.set('trackingId', 'fake-tracking-id');

  const { container } = render(StartRecipe, {
    recipeId: 'dummy-recipe-id',
  });

  return await vi.waitFor(() => {
    const input = container.querySelector('input[name="select-model"][type="hidden"]');
    if (!input) throw new Error('input not found');
    expect(JSON.parse((input as HTMLInputElement).value).label).toBe(fakeRecommendedModel.name);
  });
});
