/**********************************************************************
 * Copyright (C) 2023 Red Hat, Inc.
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

import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeAll, expect, test, vi } from 'vitest';

import DropDownMenu from './DropDownMenu.svelte';
class ResizeObserver {
  observe = vi.fn();
  disconnect = vi.fn();
  unobserve = vi.fn();
}

beforeAll(() => {
  (window as any).ResizeObserver = ResizeObserver;
});

test('Expect the onBeforeToggle function to be called when the menu is clicked', async () => {
  const onBeforeToggleMock = vi.fn();
  render(DropDownMenu, {
    onBeforeToggle: onBeforeToggleMock,
  });

  const toggleMenuButton = screen.getByRole('button', { name: 'kebab menu' });
  expect(toggleMenuButton).not.toHaveClass('mr-2');
  expect(toggleMenuButton).toBeInTheDocument();
  await fireEvent.click(toggleMenuButton);

  expect(onBeforeToggleMock).toHaveBeenCalled();
});
