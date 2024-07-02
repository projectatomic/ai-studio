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
import { render, screen, waitFor, within } from '@testing-library/svelte';
import { beforeEach, expect, test, vi } from 'vitest';
import * as catalogStore from '/@/stores/catalog';
import { readable } from 'svelte/store';
import type { ApplicationCatalog } from '@shared/src/models/IApplicationCatalog';
import RecipeModels from '/@/pages/RecipeModels.svelte';
import type { InferenceServer } from '@shared/src/models/IInference';

vi.mock('/@/stores/catalog', async () => {
  return {
    catalog: vi.fn(),
  };
});

vi.mock('../stores/inferenceServers', () => ({
  inferenceServers: {
    subscribe: (f: (msg: InferenceServer[]) => void) => {
      f([]);
      return () => {};
    },
  },
}));

beforeEach(() => {
  vi.resetAllMocks();
  const catalog: ApplicationCatalog = {
    recipes: [
      {
        id: 'recipe1',
        name: 'Recipe 1',
        recommended: ['model1'],
        categories: [],
        description: 'Recipe 1',
        readme: '',
        repository: 'https://podman-desktop.io',
      },
    ],
    models: [
      {
        id: 'model1',
        name: 'Model 1',
        url: 'https://podman-desktop.io',
        registry: 'Podman Desktop',
        license: 'Apache 2.0',
        description: '',
        hw: 'CPU',
        memory: 4 * 1024 * 1024 * 1024,
      },
    ],
    categories: [],
  };
  vi.mocked(catalogStore).catalog = readable(catalog);
});

test('should display model icon', async () => {
  render(RecipeModels, {
    recommended: [],
    selected: 'model1',
    setSelectedModel: vi.fn(),
    models: [
      {
        id: 'model1',
        name: 'Model 1',
        url: 'https://podman-desktop.io',
        registry: 'Podman Desktop',
        license: 'Apache 2.0',
        description: '',
        hw: 'CPU',
        memory: 4 * 1024 * 1024 * 1024,
      },
    ],
  });

  await waitFor(async () => {
    const table = screen.getByRole('table');
    expect(table).toBeInTheDocument();
    const rows = screen.queryAllByRole('rowgroup');
    expect(rows.length > 0).toBeTruthy();

    const icon = await within(rows[1]).findByRole('img');
    expect(icon).toBeDefined();
  });
});
