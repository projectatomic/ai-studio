<script lang="ts">
import { applicationStates } from '../stores/application-states';
import ColumnActions from '../lib/table/application/ColumnActions.svelte';
import ColumnStatus from '../lib/table/application/ColumnStatus.svelte';
import ColumnRecipe from '../lib/table/application/ColumnRecipe.svelte';
import ColumnModel from '../lib/table/application/ColumnModel.svelte';
import ColumnPod from '../lib/table/application/ColumnPod.svelte';
import ColumnAge from '../lib/table/application/ColumnAge.svelte';
import { router } from 'tinro';
import { onMount } from 'svelte';
import type { ApplicationState } from '@shared/src/models/IApplicationState';
import { Table, TableColumn, TableRow, NavPage } from '@podman-desktop/ui-svelte';
import TasksBanner from '/@/lib/progress/TasksBanner.svelte';

const columns: TableColumn<ApplicationState>[] = [
  new TableColumn<ApplicationState>('Status', { width: '70px', align: 'center', renderer: ColumnStatus }),
  new TableColumn<ApplicationState>('Model', { width: '3fr', renderer: ColumnModel }),
  new TableColumn<ApplicationState>('Recipe', { width: '2fr', renderer: ColumnRecipe }),
  new TableColumn<ApplicationState>('Pod', { width: '3fr', renderer: ColumnPod }),
  new TableColumn<ApplicationState>('Age', { width: '2fr', renderer: ColumnAge }),
  new TableColumn<ApplicationState>('Actions', {
    align: 'right',
    width: '120px',
    renderer: ColumnActions,
    overflow: true,
  }),
];
const row = new TableRow<ApplicationState>({});

const openApplicationCatalog = () => {
  router.goto('/recipes');
};

let data: (ApplicationState & { selected?: boolean })[];

onMount(() => {
  return applicationStates.subscribe(items => {
    data = items;
  });
});
</script>

<NavPage title="AI Apps" searchEnabled="{false}">
  <div slot="content" class="flex flex-col min-w-full min-h-full">
    <div class="min-w-full min-h-full flex-1">
      <div class="mt-4 px-5 space-y-5">
        <!-- showing running tasks -->
        <TasksBanner title="Pulling recipes" labels="{['recipe-pulling']}" />

        {#if data?.length > 0}
          <Table kind="AI App" data="{data}" columns="{columns}" row="{row}"></Table>
        {:else}
          <div class="w-full flex items-center justify-center text-[var(--pd-content-text)]">
            <div role="status">
              There is no AI App running. You may run a new AI App via the <a
                href="{'javascript:void(0);'}"
                class="underline"
                role="button"
                title="Open the catalog page"
                on:click="{openApplicationCatalog}">Recipes Catalog</a
              >.
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</NavPage>
