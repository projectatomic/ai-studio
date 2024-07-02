<script lang="ts">
import Fa from 'svelte-fa';
import { faTriangleExclamation } from '@fortawesome/free-solid-svg-icons';
import { filesize } from 'filesize';
import { studioClient } from '/@/utils/client';
import type { ContainerConnectionInfo } from '@shared/src/models/IContainerConnectionInfo';
import { Button } from '@podman-desktop/ui-svelte';

export let connectionInfo: ContainerConnectionInfo;

let title: string | undefined = '';
let description: string | undefined = '';
let actionName: string | undefined = '';
$: updateTitleDescription(connectionInfo);

function updateTitleDescription(connectionInfo: ContainerConnectionInfo) {
  if (connectionInfo.status === 'native') {
    return;
  }

  if (connectionInfo.status === 'no-machine') {
    title = 'No Podman Machine is running';
    description = 'Please start a Podman Machine before proceeding further.';
    actionName = connectionInfo.canRedirect ? 'Start now' : undefined;
    return;
  }

  if (connectionInfo.status === 'low-resources') {
    title = 'Update your Podman Machine to improve performance';

    const hasEnoughCPU = connectionInfo.cpus >= connectionInfo.cpusExpected;
    const hasEnoughMemory = connectionInfo.memoryIdle > connectionInfo.memoryExpected;

    let machineCurrentStateDescription = '';
    let machinePreferredStateDescription = '';
    if (!hasEnoughCPU) {
      machineCurrentStateDescription += `${connectionInfo.cpus} vCPUs`;
      machinePreferredStateDescription += `${connectionInfo.cpusExpected} vCPUs`;
      if (!hasEnoughMemory) {
        machineCurrentStateDescription += ` and ${filesize(connectionInfo.memoryIdle, { base: 2 })} of memory available`;
        machinePreferredStateDescription += ` and ${filesize(connectionInfo.memoryExpected, { base: 2 })} of memory`;
      }
    } else {
      machineCurrentStateDescription += `${filesize(connectionInfo.memoryIdle, { base: 2 })} of memory available`;
      machinePreferredStateDescription += `${filesize(connectionInfo.memoryExpected, { base: 2 })} of memory`;
    }

    const machineName = `${connectionInfo.name.includes('Podman Machine') ? connectionInfo.name : `Podman Machine ${connectionInfo.name}`}`;
    description = `Your ${machineName} has ${machineCurrentStateDescription}. `;

    if (connectionInfo?.canEdit) {
      description += `We recommend updating your Podman Machine to at least ${machinePreferredStateDescription} for better AI performance.`;
      actionName = connectionInfo.canRedirect ? 'Update now' : undefined;
    } else {
      description += `We recommend freeing some resources on your Podman Machine to have at least ${machinePreferredStateDescription} for better AI performance.`;
    }
    return;
  }

  title = undefined;
  description = undefined;
  actionName = undefined;
}

function executeCommand() {
  if (connectionInfo.canRedirect) {
    if (connectionInfo.status === 'low-resources' && connectionInfo.canEdit) {
      studioClient.navigateToEditConnectionProvider(connectionInfo.name);
      return;
    }
    if (connectionInfo.status == 'no-machine') {
      studioClient.navigateToResources();
    }
  }
}
</script>

{#if title && description}
  <div
    class="w-full bg-[var(--pd-content-card-bg)] text-[var(--pd-content-card-text)] border-t-[3px] border-amber-500 p-4 mt-5 shadow-inner"
    aria-label="Container connection info banner">
    <div class="flex flex-row space-x-3">
      <div class="flex">
        <Fa icon="{faTriangleExclamation}" class="text-amber-400" />
      </div>
      <div class="flex flex-col grow">
        <span class="font-medium text-sm" aria-label="title">{title}</span>
        <span class="text-sm" aria-label="description">{description}</span>
      </div>
      {#if actionName}
        <div class="flex items-center">
          <Button class="grow text-gray-500" on:click="{executeCommand}" aria-label="{actionName}">{actionName}</Button>
        </div>
      {/if}
    </div>
  </div>
{/if}
