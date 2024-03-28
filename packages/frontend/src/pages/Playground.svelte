<script lang="ts">
import { conversations } from '../stores/conversations';
import { studioClient } from '/@/utils/client';
import { isAssistantChat, isPendingChat, isUserChat, isSystemPrompt } from '@shared/src/models/IPlaygroundMessage';
import NavPage from '../lib/NavPage.svelte';
import { playgrounds } from '../stores/playgrounds-v2';
import { catalog } from '../stores/catalog';
import Button from '../lib/button/Button.svelte';
import { afterUpdate } from 'svelte';
import ContentDetailsLayout from '../lib/ContentDetailsLayout.svelte';
import RangeInput from '../lib/RangeInput.svelte';
import Fa from 'svelte-fa';

import ChatMessage from '../lib/conversation/ChatMessage.svelte';
import SystemPromptBanner from '/@/lib/conversation/SystemPromptBanner.svelte';
import { inferenceServers } from '/@/stores/inferenceServers';

export let playgroundId: string;
let prompt: string;
let sendEnabled = false;
let scrollable: Element;
let lastIsUserMessage = false;
let errorMsg = '';

// settings
let temperature = 0.8;
let max_tokens = -1;
let top_p = 0.5;

$: conversation = $conversations.find(conversation => conversation.id === playgroundId);
$: messages = conversation?.messages.filter(message => !isSystemPrompt(message)) ?? [];
$: playground = $playgrounds.find(playground => playground.id === playgroundId);
$: model = $catalog.models.find(model => model.id === playground?.modelId);
$: {
  if (conversation?.messages.length) {
    const latest = conversation.messages[conversation.messages.length - 1];
    if (isSystemPrompt(latest) || (isAssistantChat(latest) && !isPendingChat(latest))) {
      sendEnabled = true;
    }
    lastIsUserMessage = isUserChat(latest);
  } else {
    sendEnabled = true;
  }
}
$: server = $inferenceServers.find(is => playground && is.models.map(mi => mi.id).includes(playground?.modelId));
function askPlayground() {
  errorMsg = '';
  sendEnabled = false;
  studioClient
    .submitPlaygroundMessage(playgroundId, prompt, {
      temperature,
      max_tokens,
      top_p,
    })
    .catch((err: unknown) => {
      errorMsg = String(err);
      sendEnabled = true;
    });
  prompt = '';
}

afterUpdate(() => {
  if (!conversation?.messages.length) {
    return;
  }
  const latest = conversation.messages[conversation.messages.length - 1];
  if (isUserChat(latest) || (isAssistantChat(latest) && isPendingChat(latest))) {
    scrollToBottom(scrollable);
  }
});

async function scrollToBottom(element: Element) {
  element.scroll?.({ top: element.scrollHeight, behavior: 'smooth' });
}

function onRegenerate(messageId: string): void {
  errorMsg = '';
  sendEnabled = false;
  studioClient
    .requestRegeneratePlaygroundMessage(playgroundId, messageId, {
      temperature,
      max_tokens,
      top_p,
    })
    .catch((err: unknown) => {
      errorMsg = String(err);
      sendEnabled = true;
    });
}

function isHealthy(status?: string, health?: string): boolean {
  return status === 'running' && (!health || health === 'healthy');
}

function getStatusColor(status?: string, health?: string): string {
  return isHealthy(status, health) ? 'bg-green-600' : 'bg-gray-900';
}

function getStatusText(status?: string, health?: string): string {
  switch (status) {
    case 'running':
      switch (health) {
        case 'healthy':
          return 'Model Service running';
        case 'starting':
          return 'Model Service starting';
        default:
          return 'Model Service not running';
      }
    default:
      return 'Model Service not running';
  }
}

function getSendPromptTitle(sendEnabled: boolean, status?: string, health?: string): string | undefined {
  if (!isHealthy(status, health)) {
    return getStatusText(status, health);
  } else if (!sendEnabled) {
    return 'Please wait, assistant is replying';
  }
  return undefined;
}
</script>

{#if playground}
  <NavPage
    lastPage="{{ name: 'Playgrounds', path: '/playgrounds' }}"
    title="{playground?.name}"
    searchEnabled="{false}"
    contentBackground="bg-charcoal-500">
    <svelte:fragment slot="subtitle">{model?.name}</svelte:fragment>
    <svelte:fragment slot="additional-actions">
      <div role="status" class="flex items-center bg-charcoal-500 p-1 rounded-md">
        <div class="w-2 h-2 {getStatusColor(server?.status, server?.health?.Status)} rounded-full mx-1"></div>
        <span class="text-xs capitalize mr-1">
          {getStatusText(server?.status, server?.health?.Status)}
        </span>
      </div>
    </svelte:fragment>
    <svelte:fragment slot="content">
      <div class="flex flex-col w-full h-full">
        <div class="h-full overflow-auto" bind:this="{scrollable}">
          <ContentDetailsLayout detailsTitle="Settings" detailsLabel="settings">
            <svelte:fragment slot="content">
              <div class="flex flex-col w-full h-full">
                <div aria-label="conversation" class="w-full h-full">
                  {#if conversation}
                    <!-- Show a banner for the system prompt -->
                    {#key conversation.messages.length}
                      <SystemPromptBanner conversation="{conversation}" />
                    {/key}
                    <!-- show all message except the sytem prompt -->
                    <ul>
                      {#each messages as message}
                        <li>
                          <ChatMessage disabled="{!sendEnabled}" onRegenerate="{onRegenerate}" message="{message}" />
                        </li>
                      {/each}
                    </ul>
                  {/if}
                </div>
              </div>
            </svelte:fragment>
            <svelte:fragment slot="details">
              <div class="text-gray-800 text-xs">Next prompt will use these settings</div>
              <div class="bg-charcoal-600 w-full rounded-md text-xs p-4">
                <div class="mb-4 flex flex-col">Model Parameters</div>
                <div class="flex flex-col space-y-4">
                  <RangeInput name="temperature" min="0" max="2" step="0.1" bind:value="{temperature}" />
                  <RangeInput name="max tokens" min="-1" max="32768" step="1" bind:value="{max_tokens}" />
                  <RangeInput name="top-p" min="0" max="1" step="0.1" bind:value="{top_p}" />
                </div>
              </div>
            </svelte:fragment>
          </ContentDetailsLayout>
        </div>
        {#if errorMsg}
          <div class="text-red-500 text-sm p-2">{errorMsg}</div>
        {/if}
        <div class="flex flex-row flex-none w-full">
          <textarea
            aria-label="prompt"
            bind:value="{prompt}"
            rows="2"
            class="w-full p-2 outline-none text-sm bg-charcoal-800 rounded-sm text-gray-700 placeholder-gray-700"
            placeholder="Type your prompt here"></textarea>

          <div class="flex-none text-right m-4">
            <Button
              inProgress="{!sendEnabled}"
              disabled="{!isHealthy(server?.status, server?.health?.Status)}"
              on:click="{() => askPlayground()}"
              title="{getSendPromptTitle(sendEnabled, server?.status, server?.health?.Status)}">Send prompt</Button>
          </div>
        </div>
      </div>
    </svelte:fragment>
  </NavPage>
{/if}
