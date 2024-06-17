<script lang="ts">
import { studioClient } from '/@/utils/client';
import { DetailsPage } from '@podman-desktop/ui-svelte';
import Card from '/@/lib/Card.svelte';
import MarkdownRenderer from '/@/lib/markdown/MarkdownRenderer.svelte';
import { getIcon } from '/@/utils/categoriesUtils';
import { catalog } from '/@/stores/catalog';
import RecipeDetails from '/@/lib/RecipeDetails.svelte';
import ContentDetailsLayout from '../lib/ContentDetailsLayout.svelte';
import { router } from 'tinro';
import { faRocket } from '@fortawesome/free-solid-svg-icons';
import { Button } from '@podman-desktop/ui-svelte';
import Fa from 'svelte-fa';

export let recipeId: string;

// The recipe model provided
$: recipe = $catalog.recipes.find(r => r.id === recipeId);
$: categories = $catalog.categories;

// Send recipe info to telemetry
let recipeTelemetry: string | undefined = undefined;
$: if (recipe && recipe.id !== recipeTelemetry) {
  recipeTelemetry = recipe.id;
  studioClient.telemetryLogUsage('recipe.open', { 'recipe.id': recipe.id, 'recipe.name': recipe.name });
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
  <svelte:fragment slot="additional-actions">
    <Button on:click="{() => router.goto(`/recipe/${recipeId}/start`)}" icon="{faRocket}" aria-label="Start recipe"
      >Start</Button>
  </svelte:fragment>
  <svelte:fragment slot="content">
    <ContentDetailsLayout detailsTitle="AI App Details" detailsLabel="application details">
      <svelte:fragment slot="content">
        <MarkdownRenderer source="{recipe?.readme}" />
      </svelte:fragment>
      <svelte:fragment slot="details">
        <RecipeDetails recipeId="{recipeId}" />
      </svelte:fragment>
    </ContentDetailsLayout>
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
