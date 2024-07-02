<script lang="ts">
import { studioClient } from '/@/utils/client';
import { Tab, DetailsPage } from '@podman-desktop/ui-svelte';
import Route from '/@/Route.svelte';
import Card from '/@/lib/Card.svelte';
import MarkdownRenderer from '/@/lib/markdown/MarkdownRenderer.svelte';
import { getIcon } from '/@/utils/categoriesUtils';
import RecipeModels from './RecipeModels.svelte';
import { catalog } from '/@/stores/catalog';
import RecipeDetails from '/@/lib/RecipeDetails.svelte';
import ContentDetailsLayout from '../lib/ContentDetailsLayout.svelte';
import type { ContainerConnectionInfo } from '@shared/src/models/IContainerConnectionInfo';
import ContainerConnectionStatusInfo from '../lib/notification/ContainerConnectionStatusInfo.svelte';
import { modelsInfo } from '../stores/modelsInfo';
import { checkContainerConnectionStatus } from '../utils/connectionUtils';
import { router } from 'tinro';
import { InferenceType } from '@shared/src/models/IInference';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import Fa from 'svelte-fa';

export let recipeId: string;

// The recipe model provided
$: recipe = $catalog.recipes.find(r => r.id === recipeId);
$: categories = $catalog.categories;

// model selected to start the recipe
let selectedModelId: string;
$: selectedModelId = recipe?.recommended && recipe.recommended.length > 0 ? recipe?.recommended?.[0] : '';

let connectionInfo: ContainerConnectionInfo | undefined;
$: if ($modelsInfo && selectedModelId) {
  checkContainerConnectionStatus($modelsInfo, selectedModelId, 'recipe')
    .then(value => (connectionInfo = value))
    .catch((e: unknown) => console.log(String(e)));
}

let models: ModelInfo[];
$: models = $catalog.models.filter(
  model => (model.backend ?? InferenceType.NONE) === (recipe?.backend ?? InferenceType.NONE),
);

// Send recipe info to telemetry
let recipeTelemetry: string | undefined = undefined;
$: if (recipe && recipe.id !== recipeTelemetry) {
  recipeTelemetry = recipe.id;
  studioClient.telemetryLogUsage('recipe.open', { 'recipe.id': recipe.id, 'recipe.name': recipe.name });
}

function setSelectedModel(modelId: string) {
  selectedModelId = modelId;
  studioClient.telemetryLogUsage('recipe.select-model', { 'recipe.id': recipe?.id, 'model.id': modelId });
}

export function goToUpPage(): void {
  router.goto('/recipes');
}
</script>

<DetailsPage
  title="{recipe?.name || ''}"
  breadcrumbLeftPart="Recipes"
  breadcrumbRightPart="{recipe?.name || ''}"
  breadcrumbTitle="Go back to Recipes"
  on:close="{goToUpPage}"
  on:breadcrumbClick="{goToUpPage}">
  <svelte:fragment slot="icon">
    <div class="rounded-full w-8 h-8 flex items-center justify-center">
      <Fa size="1.125x" class="text-[var(--pd-content-header-icon)]" icon="{getIcon(recipe?.icon)}" />
    </div>
  </svelte:fragment>
  <svelte:fragment slot="tabs">
    <Tab title="Summary" url="/recipe/{recipeId}" selected="{$router.path === `/recipe/${recipeId}`}" />
    <Tab title="Models" url="/recipe/{recipeId}/models" selected="{$router.path === `/recipe/${recipeId}/models`}" />
  </svelte:fragment>
  <svelte:fragment slot="content">
    <div class="bg-[var(--pd-content-bg)] h-full overflow-y-auto">
      <ContentDetailsLayout detailsTitle="AI App Details" detailsLabel="application details">
        <svelte:fragment slot="header">
          {#if connectionInfo}
            <div class="px-4">
              <ContainerConnectionStatusInfo connectionInfo="{connectionInfo}" />
            </div>
          {/if}
        </svelte:fragment>
        <svelte:fragment slot="content">
          <Route path="/">
            <MarkdownRenderer source="{recipe?.readme}" />
          </Route>
          <Route path="/models">
            <RecipeModels
              models="{models}"
              selected="{selectedModelId}"
              recommended="{recipe?.recommended ?? []}"
              setSelectedModel="{setSelectedModel}" />
          </Route>
        </svelte:fragment>
        <svelte:fragment slot="details">
          <RecipeDetails recipeId="{recipeId}" modelId="{selectedModelId}" />
        </svelte:fragment>
      </ContentDetailsLayout>
    </div>
  </svelte:fragment>
  <svelte:fragment slot="subtitle">
    <div class="mt-2">
      {#each recipe?.categories || [] as categoryId}
        <Card
          title="{categories.find(category => category.id === categoryId)?.name || '?'}"
          classes="bg-charcoal-800 p-1 text-xs w-fit" />
      {/each}
    </div>
  </svelte:fragment>
</DetailsPage>
