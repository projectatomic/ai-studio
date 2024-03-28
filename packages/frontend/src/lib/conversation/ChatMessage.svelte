<script lang="ts">
import {
  type AssistantChat,
  type ChatMessage,
  isAssistantChat,
  isPendingChat,
  isSystemPrompt,
  isUserChat,
} from '@shared/src/models/IPlaygroundMessage';
import Fa from 'svelte-fa';
import { faRotateRight } from '@fortawesome/free-solid-svg-icons';

export let message: ChatMessage;
export let disabled: boolean;
export let onRegenerate: (messageId: string) => void;

const roles = {
  system: 'System prompt',
  user: 'User',
  assistant: 'Assistant',
};

function getMessageParagraphs(message: ChatMessage): string[] {
  if (isAssistantChat(message)) {
    if (!isPendingChat(message)) {
      return message.content?.split('\n') ?? [];
    } else {
      return message.choices
        .map(choice => choice.content)
        .join('')
        .split('\n');
    }
  } else if (isUserChat(message) || isSystemPrompt(message)) {
    return message.content?.split('\n') ?? [];
  }
  return [];
}

function elapsedTime(msg: AssistantChat): string {
  if (isPendingChat(msg)) {
    return ((Date.now() - msg.timestamp) / 1000).toFixed(1);
  } else if (!!msg.completed) {
    return ((msg.completed - msg.timestamp) / 1000).toFixed(1);
  } else {
    // should not happen
    return '';
  }
}
</script>

<div>
  <div class="text-lg" class:text-right="{isAssistantChat(message)}">
    {roles[message.role]}
  </div>
  <div
    class="p-4 rounded-md"
    class:bg-charcoal-400="{isUserChat(message)}"
    class:bg-charcoal-800="{isSystemPrompt(message)}"
    class:bg-charcoal-900="{isAssistantChat(message)}"
    class:ml-8="{isAssistantChat(message)}"
    class:mr-8="{isUserChat(message)}">
    {#each getMessageParagraphs(message) as paragraph}
      <p>{paragraph}</p>
    {/each}
  </div>
  {#if isAssistantChat(message)}
    <div class="flex w-full justify-end items-center gap-x-2">
      <button
        class:text-gray-900="{disabled}"
        disabled="{disabled}"
        on:click="{() => onRegenerate(message.id)}"
        title="Regenerate"><Fa icon="{faRotateRight}" /></button>
      <div class="text-sm text-gray-400 text-right" aria-label="elapsed">
        {elapsedTime(message)} s
      </div>
    </div>
  {/if}
  <div></div>
</div>
