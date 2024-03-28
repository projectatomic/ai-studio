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
import type { Disposable, Webview } from '@podman-desktop/api';
import type { InferenceManager } from './inference/inferenceManager';
import OpenAI from 'openai';
import type { ChatCompletionChunk, ChatCompletionMessageParam } from 'openai/src/resources/chat/completions';
import type { ModelOptions } from '@shared/src/models/IModelOptions';
import type { Stream } from 'openai/streaming';
import { ConversationRegistry } from '../registries/conversationRegistry';
import type { Conversation, PendingChat, SystemPrompt, UserChat } from '@shared/src/models/IPlaygroundMessage';
import { isAssistantChat, isSystemPrompt } from '@shared/src/models/IPlaygroundMessage';
import type { PlaygroundV2 } from '@shared/src/models/IPlaygroundV2';
import { Publisher } from '../utils/Publisher';
import { Messages } from '@shared/Messages';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import { withDefaultConfiguration } from '../utils/inferenceUtils';
import { getRandomString } from '../utils/randomUtils';
import type { TaskRegistry } from '../registries/TaskRegistry';
import type { InferenceServer } from '@shared/src/models/IInference';

export class PlaygroundV2Manager extends Publisher<PlaygroundV2[]> implements Disposable {
  #playgrounds: Map<string, PlaygroundV2>;
  #conversationRegistry: ConversationRegistry;
  #playgroundCounter = 0;
  #UIDcounter: number;

  constructor(
    webview: Webview,
    private inferenceManager: InferenceManager,
    private taskRegistry: TaskRegistry,
  ) {
    super(webview, Messages.MSG_PLAYGROUNDS_V2_UPDATE, () => this.getPlaygrounds());
    this.#playgrounds = new Map();
    this.#conversationRegistry = new ConversationRegistry(webview);
    this.#UIDcounter = 0;
  }

  deletePlayground(conversationId: string): void {
    this.#conversationRegistry.deleteConversation(conversationId);
    this.#playgrounds.delete(conversationId);
    this.notify();
  }

  async requestCreatePlayground(name: string, model: ModelInfo, systemPrompt?: string): Promise<string> {
    const trackingId: string = getRandomString();
    const task = this.taskRegistry.createTask('Creating Playground environment', 'loading', {
      trackingId: trackingId,
    });

    this.createPlayground(name, model, systemPrompt, trackingId)
      .then((playgroundId: string) => {
        this.taskRegistry.updateTask({
          ...task,
          state: 'success',
          labels: {
            ...task.labels,
            playgroundId,
          },
        });
      })
      .catch((err: unknown) => {
        const tasks = this.taskRegistry.getTasksByLabels({
          trackingId: trackingId,
        });
        // Filter the one no in loading state
        tasks
          .filter(t => t.state === 'loading' && t.id !== task.id)
          .forEach(t => {
            this.taskRegistry.updateTask({
              ...t,
              state: 'error',
            });
          });
        // Update the main task
        this.taskRegistry.updateTask({
          ...task,
          state: 'error',
          error: `Something went wrong while trying to create a playground environment ${String(err)}.`,
        });
      });
    return trackingId;
  }

  async createPlayground(
    name: string,
    model: ModelInfo,
    systemPrompt: string | undefined,
    trackingId: string,
  ): Promise<string> {
    const id = `${this.#playgroundCounter++}`;

    if (!name) {
      name = this.getFreeName();
    }

    this.#conversationRegistry.createConversation(id);

    if (systemPrompt !== undefined && systemPrompt.length > 0) {
      this.#conversationRegistry.submit(id, {
        content: systemPrompt,
        role: 'system',
        id: this.getUniqueId(),
        timestamp: Date.now(),
      } as SystemPrompt);
    }

    // create/start inference server if necessary
    const servers = this.inferenceManager.getServers();
    const server = servers.find(s => s.models.map(mi => mi.id).includes(model.id));
    if (!server) {
      await this.inferenceManager.createInferenceServer(
        await withDefaultConfiguration({
          modelsInfo: [model],
        }),
        trackingId,
      );
    } else if (server.status === 'stopped') {
      await this.inferenceManager.startInferenceServer(server.container.containerId);
    }

    this.#playgrounds.set(id, {
      id,
      name,
      modelId: model.id,
    });
    this.notify();

    return id;
  }

  getPlaygrounds(): PlaygroundV2[] {
    return Array.from(this.#playgrounds.values());
  }

  private getUniqueId(): string {
    return `playground-${++this.#UIDcounter}`;
  }

  /**
   * Given a conversation, update the system prompt.
   * If none exists, it will create one, otherwise it will replace the content with the new one
   * @param conversationId the conversation id to set the system id
   * @param content the new system prompt to use
   */
  setSystemPrompt(conversationId: string, content: string | undefined): void {
    const conversation = this.#conversationRegistry.get(conversationId);
    if (conversation === undefined) throw new Error(`Conversation with id ${conversationId} does not exists.`);

    if (conversation.messages.length === 0) {
      this.#conversationRegistry.submit(conversationId, {
        role: 'system',
        content,
        timestamp: Date.now(),
        id: this.getUniqueId(),
      } as SystemPrompt);
      this.notify();
    } else if (conversation.messages.length === 1 && isSystemPrompt(conversation.messages[0])) {
      if (content !== undefined && content.length > 0) {
        this.#conversationRegistry.update(conversationId, conversation.messages[0].id, {
          content,
        });
      } else {
        this.#conversationRegistry.removeMessages(conversationId, conversation.messages[0].id);
      }
    } else {
      throw new Error('Cannot change system prompt on started conversation.');
    }
  }

  async regenerate(conversationId: string, messageId: string, options?: ModelOptions): Promise<void> {
    const playground = this.#playgrounds.get(conversationId);
    if (playground === undefined) throw new Error('Playground not found.');

    const conversation = this.#conversationRegistry.get(conversationId);
    if (conversation === undefined) throw new Error(`conversation with id ${conversationId} does not exist.`);

    const messageIndex = conversation.messages.findIndex(message => message.id === messageId);
    if (messageIndex === -1) throw new Error(`Cannot find a message with id ${messageId} to regenerate.`);

    if (!isAssistantChat(conversation.messages[messageIndex])) throw new Error('Only assistant chat can be replayed.');

    this.#conversationRegistry.removeMessages(
      conversationId,
      ...conversation.messages.filter((_message, index) => index >= messageIndex).map(message => message.id),
    );

    this.execute(conversationId, this.getCorrespondingServer(playground.modelId), playground.modelId, options);
  }

  /**
   * @param conversationId
   * @param userInput the user input
   * @param options the model configuration
   */
  async submit(conversationId: string, userInput: string, options?: ModelOptions): Promise<void> {
    const playground = this.#playgrounds.get(conversationId);
    if (playground === undefined) throw new Error('Playground not found.');

    const server = this.getCorrespondingServer(playground.modelId);

    const conversation = this.#conversationRegistry.get(conversationId);
    if (conversation === undefined) throw new Error(`conversation with id ${conversationId} does not exist.`);

    this.#conversationRegistry.submit(conversationId, {
      content: userInput,
      options: options,
      role: 'user',
      id: this.getUniqueId(),
      timestamp: Date.now(),
    } as UserChat);

    this.execute(conversationId, server, playground.modelId, options);
  }

  private getCorrespondingServer(modelId: string): InferenceServer {
    const servers = this.inferenceManager.getServers();
    const server = servers.find(s => s.models.map(mi => mi.id).includes(modelId));
    if (server === undefined) throw new Error('Inference server not found.');

    if (server.status !== 'running') throw new Error('Inference server is not running.');

    if (server.health?.Status !== 'healthy')
      throw new Error(`Inference server is not healthy, currently status: ${server.health.Status}.`);
    return server;
  }

  private execute(conversationId: string, server: InferenceServer, modelId: string, options?: ModelOptions): void {
    const client = new OpenAI({
      baseURL: `http://localhost:${server.connection.port}/v1`,
      apiKey: 'dummy',
    });

    const modelInfo = server.models.find(model => model.id === modelId);
    if (modelInfo === undefined)
      throw new Error(
        `modelId '${modelId}' is not available on the inference server, valid model ids are: ${server.models.map(model => model.id).join(', ')}.`,
      );

    client.chat.completions
      .create({
        messages: this.getFormattedMessages(conversationId),
        stream: true,
        model: modelInfo.file.file,
        ...options,
      })
      .then(response => {
        // process stream async
        this.processStream(conversationId, response).catch((err: unknown) => {
          console.error('Something went wrong while processing stream', err);
        });
      })
      .catch((err: unknown) => {
        console.error('Something went wrong while creating model reponse', err);
      });
  }

  /**
   * Given a Stream from the OpenAI library update and notify the publisher
   * @param conversationId
   * @param stream
   */
  private async processStream(conversationId: string, stream: Stream<ChatCompletionChunk>): Promise<void> {
    const messageId = this.getUniqueId();
    this.#conversationRegistry.submit(conversationId, {
      role: 'assistant',
      choices: [],
      completed: undefined,
      id: messageId,
      timestamp: Date.now(),
    } as PendingChat);

    for await (const chunk of stream) {
      this.#conversationRegistry.appendChoice(conversationId, messageId, {
        content: chunk.choices[0]?.delta?.content || '',
      });
    }

    this.#conversationRegistry.completeMessage(conversationId, messageId);
  }

  /**
   * Transform the ChatMessage interface to the OpenAI ChatCompletionMessageParam
   * @private
   */
  private getFormattedMessages(conversationId: string): ChatCompletionMessageParam[] {
    return this.#conversationRegistry.get(conversationId).messages.map(
      message =>
        ({
          name: undefined,
          ...message,
        }) as ChatCompletionMessageParam,
    );
  }

  getConversations(): Conversation[] {
    return this.#conversationRegistry.getAll();
  }

  dispose(): void {
    this.#conversationRegistry.dispose();
  }

  getFreeName(): string {
    let i = 0;
    let name: string;
    do {
      name = `playground ${++i}`;
    } while (this.getPlaygrounds().find(p => p.name === name));
    return name;
  }
}
