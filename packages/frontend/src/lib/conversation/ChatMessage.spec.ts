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
import { expect, test, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import ChatMessage from '/@/lib/conversation/ChatMessage.svelte';
import type { AssistantChat } from '@shared/src/models/IPlaygroundMessage';

beforeEach(() => {
  vi.resetAllMocks();
});

test('assistant message should show elapsed time', () => {
  render(ChatMessage, {
    disabled: false,
    onRegenerate: vi.fn(),
    message: {
      id: 'dummyId',
      content: 'dummyContent',
      role: 'assistant',
      timestamp: 5000,
      choices: [],
      completed: 10000,
    } as AssistantChat,
  });

  const elapsed = screen.getByLabelText('elapsed');
  expect(elapsed).toBeInTheDocument();
  expect(elapsed.textContent).toBe('5.0 s');
});

test('clicking on regenerate should call onRegenerate', async () => {
  const onRegenerateMock = vi.fn();

  render(ChatMessage, {
    disabled: false,
    onRegenerate: onRegenerateMock,
    message: {
      id: 'dummyId',
      content: 'dummyContent',
      role: 'assistant',
      timestamp: 0,
      choices: [],
    } as AssistantChat,
  });

  const regenerateBtn = screen.getByTitle('Regenerate');
  expect(regenerateBtn).toBeInTheDocument();

  await fireEvent.click(regenerateBtn);

  expect(onRegenerateMock).toHaveBeenCalledWith('dummyId');
});
