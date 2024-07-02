<script lang="ts">
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import { faDownload, faRocket, faTrash } from '@fortawesome/free-solid-svg-icons';
import { faFolderOpen } from '@fortawesome/free-solid-svg-icons';
import ListItemButtonIcon from '../../button/ListItemButtonIcon.svelte';
import { studioClient } from '/@/utils/client';
import { router } from 'tinro';
import { onMount } from 'svelte';
import { inferenceServers } from '/@/stores/inferenceServers';
export let object: ModelInfo;

let inUse: boolean = false;
$: inUse;

function deleteModel() {
  studioClient.requestRemoveLocalModel(object.id).catch(err => {
    console.error(`Something went wrong while trying to delete model ${String(err)}.`);
  });
}

function openModelFolder() {
  if (object && object.file) {
    studioClient.openFile(object.file.path);
  }
}

function downloadModel() {
  if (object && object.file === undefined) {
    studioClient.downloadModel(object.id).catch((err: unknown) => {
      console.error(`Something went wrong while trying to download model ${object.id}`, err);
    });
  }
}

function createModelService() {
  router.goto('/service/create');
  router.location.query.replace({ 'model-id': object.id });
}

onMount(() => {
  return inferenceServers.subscribe(servers => {
    inUse = servers.some(server => server.models.some(model => model.id === object.id));
  });
});
</script>

{#if object.file !== undefined}
  <ListItemButtonIcon
    icon="{faRocket}"
    title="Create Model Service"
    enabled="{!object.state}"
    onClick="{() => createModelService()}" />
  <ListItemButtonIcon
    icon="{faFolderOpen}"
    onClick="{() => openModelFolder()}"
    title="Open Model Folder"
    enabled="{!object.state}" />
  <span>{inUse}</span>
  <ListItemButtonIcon
    icon="{faTrash}"
    onClick="{deleteModel}"
    title="Delete Model"
    enabled="{!inUse && !object.state}" />
{:else}
  <ListItemButtonIcon icon="{faDownload}" onClick="{downloadModel}" title="Download Model" enabled="{!object.state}" />
{/if}
