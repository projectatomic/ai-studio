<script lang="ts">
import type { IconDefinition } from '@fortawesome/free-solid-svg-icons';
import Fa from 'svelte-fa';
import { DropdownMenu } from '@podman-desktop/ui-svelte';
export let title: string;
export let icon: IconDefinition;
export let hidden = false;
export let enabled: boolean = true;
export let onClick: () => void = () => {};
export let menu = false;
export let detailed = false;
export let inProgress = false;
export let iconOffset = '';
export let tooltip: string = '';

let positionLeftClass = 'left-1';
if (detailed) positionLeftClass = 'left-2';
let positionTopClass = 'top-1';
if (detailed) positionTopClass = '[0.2rem]';

const buttonDetailedClass =
  'text-[var(--pd-action-button-details-text)] bg-[var(--pd-action-button-details-bg)] hover:text-[var(--pd-action-button-details-hover-text)] font-medium rounded-lg text-sm inline-flex items-center px-3 py-2 text-center';
const buttonDetailedDisabledClass =
  'text-[var(--pd-action-button-details-disabled-text)] bg-[var(--pd-action-button-details-disabled-bg)] font-medium rounded-lg text-sm inline-flex items-center px-3 py-2 text-center';
const buttonClass =
  'm-0.5 text-[var(--pd-action-button-text)] hover:bg-[var(--pd-action-button-hover-bg)] hover:text-[var(--pd-action-button-hover-text)] font-medium rounded-full inline-flex items-center px-2 py-2 text-center';
const buttonDisabledClass =
  'm-0.5 text-[var(--pd-action-button-disabled-text)] font-medium rounded-full inline-flex items-center px-2 py-2 text-center';

$: handleClick =
  enabled && !inProgress
    ? onClick
    : () => {
        console.log('==> 1');
      };
$: styleClass = detailed
  ? enabled && !inProgress
    ? buttonDetailedClass
    : buttonDetailedDisabledClass
  : enabled && !inProgress
    ? buttonClass
    : buttonDisabledClass;
</script>

{#if menu}
  <!-- enabled menu -->
  <DropdownMenu.Item
    title="{title}"
    tooltip="{tooltip}"
    icon="{icon}"
    enabled="{enabled}"
    hidden="{hidden}"
    onClick="{handleClick}" />
{:else}
  <button
    title="{title}"
    aria-label="{title}"
    on:click="{handleClick}"
    class="{styleClass} relative"
    class:disabled="{inProgress}"
    class:hidden="{hidden}">
    <Fa class="h-4 w-4 {iconOffset}" icon="{icon}" />
    <div
      aria-label="spinner"
      class="w-6 h-6 rounded-full animate-spin border border-solid border-[var(--pd-action-button-spinner)] border-t-transparent absolute {positionTopClass} {positionLeftClass}"
      class:hidden="{!inProgress}">
    </div>
  </button>
{/if}
