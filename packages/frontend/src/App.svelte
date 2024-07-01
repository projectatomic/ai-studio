<script lang="ts">
import './app.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import { router } from 'tinro';
import Route from '/@/Route.svelte';
import Navigation from '/@/lib/Navigation.svelte';
import Dashboard from '/@/pages/Dashboard.svelte';
import Recipes from '/@/pages/Recipes.svelte';
import Applications from './pages/Applications.svelte';
import Preferences from '/@/pages/Preferences.svelte';
import Models from '/@/pages/Models.svelte';
import Recipe from '/@/pages/Recipe.svelte';
import Model from './pages/Model.svelte';
import { onMount } from 'svelte';
import { getRouterState } from '/@/utils/client';
import CreateService from '/@/pages/CreateService.svelte';
import Services from '/@/pages/InferenceServers.svelte';
import ServiceDetails from '/@/pages/InferenceServerDetails.svelte';
import Playgrounds from './pages/Playgrounds.svelte';
import Playground from './pages/Playground.svelte';
import PlaygroundCreate from './pages/PlaygroundCreate.svelte';
import ImportModels from './pages/ImportModel.svelte';
import StartRecipe from '/@/pages/StartRecipe.svelte';

router.mode.hash();

let isMounted = false;

onMount(() => {
  // Load router state on application startup
  const state = getRouterState();
  router.goto(state.url);
  isMounted = true;
});
</script>

<Route path="/*" isAppMounted="{isMounted}" let:meta>
  <main class="flex flex-col w-screen h-screen overflow-hidden bg-[var(--pd-content-bg)]">
    <div class="flex flex-row w-full h-full overflow-hidden">
      <Navigation meta="{meta}" />

      <!-- Dashboard -->
      <Route path="/">
        <Dashboard />
      </Route>

      <!-- Recipes Catalog -->
      <Route path="/recipes">
        <Recipes />
      </Route>

      <!-- Applications -->
      <Route path="/applications">
        <Applications />
      </Route>

      <!-- Playgrounds -->
      <Route path="/playgrounds">
        <Playgrounds />
      </Route>
      <Route path="/playground/:id/*" let:meta>
        {#if meta.params.id === 'create'}
          <PlaygroundCreate />
        {:else}
          <Playground playgroundId="{meta.params.id}" />
        {/if}
      </Route>

      <!-- Preferences -->
      <Route path="/preferences">
        <Preferences />
      </Route>

      <!-- Recipes -->
      <Route path="/recipe/:id/*" firstmatch let:meta>
        <Route path="/start">
          <StartRecipe recipeId="{meta.params.id}" />
        </Route>
        <Route path="/*">
          <Recipe recipeId="{meta.params.id}" />
        </Route>
      </Route>

      <!-- Models -->
      <Route path="/models/*" firstmatch>
        <Route path="/import">
          <ImportModels />
        </Route>
        <Route path="/*">
          <Models />
        </Route>
      </Route>

      <Route path="/model/:id/*" let:meta>
        <Model modelId="{meta.params.id}" />
      </Route>

      <!-- services -->
      <Route path="/services/*">
        <Services />
      </Route>

      <Route path="/service/:id/*" let:meta>
        {#if meta.params.id === 'create'}
          <CreateService />
        {:else}
          <ServiceDetails containerId="{meta.params.id}" />
        {/if}
      </Route>
    </div>
  </main>
</Route>
