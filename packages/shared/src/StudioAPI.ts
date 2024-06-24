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

import type { ModelCheckerInfo, ModelInfo } from './models/IModelInfo';
import type { ApplicationCatalog } from './models/IApplicationCatalog';
import type { OpenDialogOptions, TelemetryTrustedValue, Uri } from '@podman-desktop/api';
import type { ApplicationState } from './models/IApplicationState';
import type { Task } from './models/ITask';
import type { LocalRepository } from './models/ILocalRepository';
import type { InferenceServer } from './models/IInference';
import type { RequestOptions } from './models/RequestOptions';
import type { Language } from 'postman-code-generators';
import type { CreationInferenceServerOptions } from './models/InferenceServerConfig';
import type { ModelOptions } from './models/IModelOptions';
import type { Conversation } from './models/IPlaygroundMessage';
import type { LocalModelImportInfo } from './models/ILocalModelInfo';
import type { ContainerConnectionInfo } from './models/IContainerConnectionInfo';
import type { ExtensionConfiguration } from './models/IExtensionConfiguration';

export abstract class StudioAPI {
  abstract ping(): Promise<string>;
  abstract getCatalog(): Promise<ApplicationCatalog>;

  // Application related methods
  /**
   * Clone a recipe
   * @param recipeId
   */
  abstract cloneApplication(recipeId: string): Promise<void>;
  abstract pullApplication(recipeId: string, modelId: string): Promise<void>;
  abstract requestStopApplication(recipeId: string, modelId: string): Promise<void>;
  abstract requestStartApplication(recipeId: string, modelId: string): Promise<void>;
  abstract requestRemoveApplication(recipeId: string, modelId: string): Promise<void>;
  abstract requestRestartApplication(recipeId: string, modelId: string): Promise<void>;
  abstract requestOpenApplication(recipeId: string, modelId: string): Promise<void>;
  abstract getApplicationsState(): Promise<ApplicationState[]>;

  abstract openURL(url: string): Promise<boolean>;
  abstract openFile(file: string, recipeId?: string): Promise<boolean>;
  abstract openDialog(options?: OpenDialogOptions): Promise<Uri[] | undefined>;

  /**
   * Get the information of models saved locally into the user's directory
   */
  abstract getModelsInfo(): Promise<ModelInfo[]>;
  /**
   * Delete the folder containing the model from local storage
   */
  abstract requestRemoveLocalModel(modelId: string): Promise<void>;

  abstract getModelsDirectory(): Promise<string>;

  abstract navigateToContainer(containerId: string): Promise<void>;
  abstract navigateToPod(podId: string): Promise<void>;
  abstract navigateToResources(): Promise<void>;
  abstract navigateToEditConnectionProvider(connectionName: string): Promise<void>;

  abstract telemetryLogUsage(eventName: string, data?: Record<string, unknown | TelemetryTrustedValue>): Promise<void>;
  abstract telemetryLogError(eventName: string, data?: Record<string, unknown | TelemetryTrustedValue>): Promise<void>;

  abstract getLocalRepositories(): Promise<LocalRepository[]>;

  abstract getTasks(): Promise<Task[]>;

  /**
   * Open the VSCode editor
   * @param directory the directory to open the editor from
   */
  abstract openVSCode(directory: string, recipeId?: string): Promise<void>;

  /**
   * Download a model from the catalog
   * @param modelId the id of the model we want to download
   */
  abstract downloadModel(modelId: string): Promise<void>;

  /**
   * Get inference servers
   */
  abstract getInferenceServers(): Promise<InferenceServer[]>;

  /**
   * Request to start an inference server
   * @param options The options to use
   *
   * @return a tracking identifier to follow progress
   */
  abstract requestCreateInferenceServer(options: CreationInferenceServerOptions): Promise<string>;

  /**
   * Start an inference server
   * @param containerId the container id of the inference server
   */
  abstract startInferenceServer(containerId: string): Promise<void>;

  /**
   * Stop an inference server
   * @param containerId the container id of the inference server
   */
  abstract stopInferenceServer(containerId: string): Promise<void>;

  /**
   * Delete an inference server container
   * @param containerIds ids of the container to delete
   */
  abstract requestDeleteInferenceServer(...containerIds: string[]): Promise<void>;

  /**
   * Return a free random port on the host machine
   */
  abstract getHostFreePort(): Promise<number>;

  /**
   * Submit a user input to the Playground linked to a conversation, model, and inference server
   * @param containerId the container id of the inference server we want to use
   * @param modelId the model to use
   * @param conversationId the conversation to input the message in
   * @param userInput the user input, e.g. 'What is the capital of France ?'
   * @param options the options for the model, e.g. temperature
   */
  abstract submitPlaygroundMessage(containerId: string, userInput: string, options?: ModelOptions): Promise<void>;

  /**
   * Given a conversation, update the system prompt.
   * If none exists, it will create one, otherwise it will replace the content with the new one
   * @param conversationId the conversation id to set the system id
   * @param content the new system prompt to use
   */
  abstract setPlaygroundSystemPrompt(conversationId: string, content: string | undefined): Promise<void>;

  /**
   * Return the conversations
   */
  abstract getPlaygroundConversations(): Promise<Conversation[]>;

  /**
   * Get the extension configuration (preferences)
   */
  abstract getExtensionConfiguration(): Promise<ExtensionConfiguration>;

  /**
   * Return the list of supported languages to generate code from.
   */
  abstract getSnippetLanguages(): Promise<Language[]>;

  /**
   * return a code snippet as a string matching the arguments and options provided
   * @param options the options for the request
   * @param language the language to use
   * @param variant the variant of the language
   */
  abstract createSnippet(options: RequestOptions, language: string, variant: string): Promise<string>;

  abstract requestCreatePlayground(name: string, model: ModelInfo): Promise<string>;

  /**
   * Delete a conversation
   * @param conversationId the conversation identifier that will be deleted
   */
  abstract requestDeleteConversation(conversationId: string): Promise<void>;

  /**
   * Delete a local path
   * @param path path to delete
   */
  abstract requestDeleteLocalRepository(path: string): Promise<void>;

  /**
   * Request the cancellation of a token
   * @param tokenId the id of the CancellationToken to cancel
   */
  abstract requestCancelToken(tokenId: number): Promise<void>;

  /**
   * Import local models selected by user
   * @param models list of local models to import
   */
  abstract importModels(models: LocalModelImportInfo[]): Promise<void>;

  /**
   * Validate a LocalModelImportInfo
   * throw an error if invalid
   */
  abstract validateLocalModel(model: LocalModelImportInfo): Promise<void>;

  /**
   * Copy the provided content to the user clipboard
   * @param content
   */
  abstract copyToClipboard(content: string): Promise<void>;

  /**
   * Check if the running podman machine is running and has enough resources to execute task
   * @param modelInfo object containing info about the model to check
   */
  abstract checkContainerConnectionStatusAndResources(modelInfo: ModelCheckerInfo): Promise<ContainerConnectionInfo>;
}
