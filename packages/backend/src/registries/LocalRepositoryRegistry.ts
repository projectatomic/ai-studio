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
import type { LocalRepository } from '@shared/src/models/ILocalRepository';
import { Messages } from '@shared/Messages';
import { type Webview, Disposable } from '@podman-desktop/api';
import { Publisher } from '../utils/Publisher';
import type { Recipe } from '@shared/src/models/IRecipe';
import fs from 'node:fs';
import path from 'node:path';

/**
 * The LocalRepositoryRegistry is responsible for keeping track of the directories where recipe are cloned
 */
export class LocalRepositoryRegistry extends Publisher<LocalRepository[]> {
  // Map path => LocalRepository
  private repositories: Map<string, LocalRepository> = new Map();

  constructor(
    webview: Webview,
    private appUserDirectory: string,
  ) {
    super(webview, Messages.MSG_LOCAL_REPOSITORY_UPDATE, () => this.getLocalRepositories());
  }

  init(recipes: Recipe[]): void {
    this.loadLocalRecipeRepositories(recipes);
  }

  register(localRepository: LocalRepository): Disposable {
    this.repositories.set(localRepository.path, localRepository);
    this.notify();

    return Disposable.create(() => {
      this.unregister(localRepository.path);
    });
  }

  unregister(path: string): void {
    this.repositories.delete(path);
    this.notify();
  }

  async deleteLocalRepository(path: string): Promise<void> {
    await fs.promises.rm(path, { recursive: true, force: true, maxRetries: 3 });
    // once it has been removed, it also update the localRepo list
    this.unregister(path);
  }

  getLocalRepositories(): LocalRepository[] {
    return Array.from(this.repositories.values());
  }

  private loadLocalRecipeRepositories(recipes: Recipe[]): void {
    recipes.forEach(recipe => {
      const recipeFolder = path.join(this.appUserDirectory, recipe.id);
      if (fs.existsSync(recipeFolder)) {
        this.register({
          path: recipeFolder,
          labels: {
            'recipe-id': recipe.id,
          },
        });
      }
    });
  }
}
