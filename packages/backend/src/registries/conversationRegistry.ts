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

import { Publisher } from '../utils/Publisher';
import type {
  AssistantChat,
  ChatMessage,
  Choice,
  Conversation,
  PendingChat,
} from '@shared/src/models/IPlaygroundMessage';
import type { Disposable, Webview } from '@podman-desktop/api';
import { Messages } from '@shared/Messages';

export class ConversationRegistry extends Publisher<Conversation[]> implements Disposable {
  #conversations: Map<string, Conversation>;
  #counter: number;

  constructor(webview: Webview) {
    super(webview, Messages.MSG_CONVERSATIONS_UPDATE, () => this.getAll());
    this.#conversations = new Map<string, Conversation>();
    this.#counter = 0;
  }

  init(): void {
    // TODO: load from file
  }

  private getUniqueId(): string {
    return `conversation-${++this.#counter}`;
  }

  /**
   * Remove messages from a conversation
   * @param conversationId
   * @param messageIds
   */
  removeMessages(conversationId: string, ...messageIds: string[]) {
    const conversation = this.#conversations.get(conversationId);

    if (conversation === undefined) {
      throw new Error(`conversation with id ${conversationId} does not exist.`);
    }

    conversation.messages = conversation.messages.filter(message => !messageIds.includes(message.id));
    this.notify();
  }

  /**
   * Utility method to update a message content in a given conversation
   * @param conversationId
   * @param messageId
   * @param message
   */
  update(conversationId: string, messageId: string, message: Partial<ChatMessage>) {
    const conversation = this.#conversations.get(conversationId);

    if (conversation === undefined) {
      throw new Error(`conversation with id ${conversationId} does not exist.`);
    }

    const messageIndex = conversation.messages.findIndex(message => message.id === messageId);
    if (messageIndex === -1)
      throw new Error(`message with id ${messageId} does not exist in conversation ${conversationId}.`);

    // Update the message with the provided content
    conversation.messages[messageIndex] = {
      ...conversation.messages[messageIndex],
      ...message,
      id: messageId, // preventing we are not updating the id
    };
    this.notify();
  }

  deleteConversation(id: string): void {
    this.#conversations.delete(id);
    this.notify();
  }

  createConversation(id: string): void {
    this.#conversations.set(id, {
      messages: [],
      id,
    });
    this.notify();
  }

  /**
   * This method will be responsible for finalizing the message by concatenating all the choices
   * @param conversationId
   * @param messageId
   */
  completeMessage(conversationId: string, messageId: string): void {
    const conversation = this.#conversations.get(conversationId);
    if (conversation === undefined) throw new Error(`conversation with id ${conversationId} does not exist.`);

    const messageIndex = conversation.messages.findIndex(message => message.id === messageId);
    if (messageIndex === -1)
      throw new Error(`message with id ${messageId} does not exist in conversation ${conversationId}.`);

    const content = ((conversation.messages[messageIndex] as PendingChat)?.choices || [])
      .map(choice => choice.content)
      .join('');

    this.update(conversationId, messageId, {
      ...conversation.messages[messageIndex],
      choices: undefined,
      role: 'assistant',
      completed: Date.now(),
      content: content,
    } as AssistantChat);
  }

  /**
   * Utility method to quickly add a choice to a given a message inside a conversation
   * @param conversationId
   * @param messageId
   * @param choice
   */
  appendChoice(conversationId: string, messageId: string, choice: Choice): void {
    const conversation = this.#conversations.get(conversationId);
    if (conversation === undefined) throw new Error(`conversation with id ${conversationId} does not exist.`);

    const messageIndex = conversation.messages.findIndex(message => message.id === messageId);
    if (messageIndex === -1)
      throw new Error(`message with id ${messageId} does not exist in conversation ${conversationId}.`);

    this.update(conversationId, messageId, {
      ...conversation.messages[messageIndex],
      choices: [...((conversation.messages[messageIndex] as PendingChat)?.choices || []), choice],
    } as PendingChat);
  }

  /**
   * Utility method to add a new Message to a given conversation
   * @param conversationId
   * @param message
   */
  submit(conversationId: string, message: ChatMessage): void {
    const conversation = this.#conversations.get(conversationId);
    if (conversation === undefined) throw new Error(`conversation with id ${conversationId} does not exist.`);

    this.#conversations.set(conversationId, {
      ...conversation,
      messages: [...conversation.messages, message],
    });
    this.notify();
  }

  dispose(): void {
    this.#conversations.clear();
  }

  get(conversationId: string): Conversation | undefined {
    return this.#conversations.get(conversationId);
  }

  getAll(): Conversation[] {
    return Array.from(this.#conversations.values());
  }
}
