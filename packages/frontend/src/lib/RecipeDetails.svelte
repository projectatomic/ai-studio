<script lang="ts">
import { faFolderOpen, faList, faTrash } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import { getDisplayName } from '/@/utils/versionControlUtils';
import TasksProgress from '/@/lib/progress/TasksProgress.svelte';
import Fa from 'svelte-fa';
import { studioClient } from '/@/utils/client';
import { catalog } from '/@/stores/catalog';
import { router } from 'tinro';
import { applicationStates } from '../stores/application-states';
import type { ApplicationState } from '@shared/src/models/IApplicationState';
import ApplicationActions from './ApplicationActions.svelte';
import Button from './button/Button.svelte';
import VSCodeIcon from '/@/lib/images/VSCodeIcon.svelte';
import { localRepositories } from '../stores/localRepositories';
import { findLocalRepositoryByRecipeId } from '/@/utils/localRepositoriesUtils';
import { tasks } from '/@/stores/tasks';
import { filterByLabel } from '/@/utils/taskUtils';
import PodIcon from '/@/lib/images/PodIcon.svelte';
import StatusIcon from '/@/lib/StatusIcon.svelte';
import type { ModelInfo } from '@shared/src/models/IModelInfo';

export let recipeId: string;
export let modelId: string;

$: appState = $applicationStates.find((app: ApplicationState) => app.recipeId === recipeId);
$: recipe = $catalog.recipes.find(r => r.id === recipeId);

$: filteredTasks = filterByLabel($tasks, {
  'recipe-id': recipeId,
  'model-id': modelId,
});

$: model = $catalog.models.find(m => m.id === modelId);
$: localPath = findLocalRepositoryByRecipeId($localRepositories, recipeId);

$: runningTask = filteredTasks.find(t => t.state === 'loading');

const onClickRepository = () => {
  if (!recipe) return;

  studioClient.openURL(recipe.repository).catch((err: unknown) => {
    console.error('Something went wrong while opening url', err);
  });
};

const openVSCode = () => {
  if (localPath) {
    studioClient.openVSCode(localPath.path);
  }
};

const navigateToPod = () => {
  if (appState?.pod.Id !== undefined) {
    studioClient.navigateToPod(appState.pod.Id);
  }
};

function findModel(id: string | undefined): ModelInfo | undefined {
  return $catalog.models.find(m => m.id === id);
}

function startApplication() {
  studioClient.pullApplication(recipeId, modelId).catch((err: unknown) => {
    console.error('Something went wrong while pulling AI App', err);
  });
}

const openLocalClone = () => {
  if (localPath) {
    studioClient.openFile(localPath.path);
  }
};

const deleteLocalClone = () => {
  if (localPath) {
    studioClient.requestDeleteLocalRepository(localPath.path);
  }
};
</script>

<div class="w-full bg-charcoal-600 rounded-md p-4">
  <div class="flex flex-row items-center">
    {#if appState && appState.pod}
      <div class="grow flex overflow-hidden whitespace-nowrap items-center">
        <a title="Navigate to Pod details" href="{'javascript:void(0);'}" on:click="{navigateToPod}">
          <StatusIcon size="{22}" status="{appState.pod.Status.toUpperCase()}" icon="{PodIcon}" />
        </a>
        <div class="ml-2 overflow-hidden">
          <div class="text-base text-gray-300 overflow-hidden text-ellipsis leading-tight">
            {appState.pod.Name}
          </div>
          <div class="text-xs text-gray-800 leading-tight">
            {appState.pod.Status.toUpperCase()}
          </div>
        </div>
      </div>
      <div class="shrink-0">
        <ApplicationActions recipeId="{recipeId}" object="{appState}" modelId="{modelId}" />
      </div>
    {:else}
      <Button inProgress="{runningTask !== undefined}" on:click="{startApplication}" class="grow text-gray-500">
        Start AI App
      </Button>
    {/if}
  </div>
  {#if filteredTasks.length > 0}
    <div class="mt-4 text-sm font-normal">
      <TasksProgress tasks="{filteredTasks}" />
    </div>
  {/if}
</div>
<div class="flex flex-col w-full space-y-4 rounded-md bg-charcoal-600 p-4">
  {#if model}
    <div class="flex flex-col space-y-2">
      <div class="flex flex-row justify-between">
        <div class="text-base">Model</div>
        <div
          class="py-0.5"
          class:hidden="{$router.path === `/recipe/${recipeId}/models`}"
          aria-label="swap model panel">
          <Button
            icon="{faList}"
            on:click="{() => router.goto(`/recipe/${recipeId}/models`)}"
            title="Go to the Models page to swap model"
            aria-label="Go to Model"
            class="h-full" />
        </div>
      </div>
      <div class="bg-charcoal-900 min-w-[200px] grow flex flex-col p-2 rounded-md space-y-3">
        <div class="flex justify-between items-center">
          <span class="text-sm" aria-label="model-selected">{model?.name}</span>
          {#if recipe?.models?.[0] === model.id}
            <i class="fas fa-star fa-xs text-gray-900" title="Recommended model"></i>
          {/if}
        </div>
        {#if model?.license}
          <div class="flex flex-row space-x-2">
            <div
              class="bg-charcoal-400 text-gray-600 text-xs font-thin px-2.5 py-0.5 rounded-md"
              aria-label="license-model">
              {model.license}
            </div>
          </div>
        {/if}
      </div>
      <div class="px-2 text-xs text-gray-700" aria-label="model-warning">
        {#if recipe?.models?.[0] === model.id}
          * This is the default, recommended model for this recipe. You can <a
            class="underline"
            href="{`/recipe/${recipeId}/models`}">swap for a different compatible model</a
          >.
        {:else}
          * The default model for this recipe is {findModel(recipe?.models?.[0])?.name}. You can
          <a class="underline" href="{`/recipe/${recipeId}/models`}"
            >swap for {findModel(recipe?.models?.[0])?.name} or a different compatible model</a
          >.
        {/if}
      </div>
    </div>
  {/if}
  <div class="flex flex-col w-full space-y-2 w-[45px]">
    <div class="text-base">Repository</div>
    <div class="cursor-pointer flex flex-col w-full space-y-2 text-nowrap">
      <button on:click="{onClickRepository}">
        <div class="flex flex-row p-0 m-0 bg-transparent items-center space-x-2">
          <Fa size="lg" icon="{faGithub}" />
          <span>{getDisplayName(recipe?.repository)}</span>
        </div>
      </button>
      {#if localPath}
        <div class="flex flex-row w-full justify-between">
          <button on:click="{openLocalClone}" aria-label="Local clone">
            <div class="flex flex-row p-0 m-0 bg-transparent items-center space-x-2">
              <Fa size="lg" icon="{faFolderOpen}" />
              <span>Local clone</span>
            </div>
          </button>
          <Button title="Delete local clone" on:click="{deleteLocalClone}" icon="{faTrash}" />
        </div>
      {/if}
    </div>
  </div>
  {#if localPath}
    <Button type="secondary" on:click="{openVSCode}" title="Open in VS Code Desktop" icon="{VSCodeIcon}"
      >Open in VSCode</Button>
  {/if}
</div>
