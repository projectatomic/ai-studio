<script lang="ts">
import { router } from 'tinro';
import NavPage from '../lib/NavPage.svelte';
import Button from '../lib/button/Button.svelte';
import Table from '../lib/table/Table.svelte';
import PlaygroundColumnModel from '../lib/table/playground/PlaygroundColumnModel.svelte';
import PlaygroundColumnName from '../lib/table/playground/PlaygroundColumnName.svelte';
import { Row, Column } from '../lib/table/table';
import type { PlaygroundV2 } from '@shared/src/models/IPlaygroundV2';
import { playgrounds } from '../stores/playgrounds-v2';
import ConversationColumnAction from '/@/lib/table/playground/ConversationColumnAction.svelte';

const columns: Column<PlaygroundV2>[] = [
  new Column<PlaygroundV2>('Name', { width: '1fr', renderer: PlaygroundColumnName }),
  new Column<PlaygroundV2>('Model', { width: '1fr', renderer: PlaygroundColumnModel }),
  new Column<PlaygroundV2>('Actions', { width: '40px', renderer: ConversationColumnAction }),
];
const row = new Row<PlaygroundV2>({});

function createNewPlayground() {
  router.goto('/playground/create');
}
</script>

<NavPage
  lastPage="{{ name: 'Playgrounds', path: '/playgrounds' }}"
  title="Playground Environments"
  searchEnabled="{false}">
  <svelte:fragment slot="additional-actions">
    <Button on:click="{() => createNewPlayground()}">New Playground</Button>
  </svelte:fragment>
  <svelte:fragment slot="content">
    <div slot="content" class="flex flex-col min-w-full">
      <div class="min-w-full flex-1">
        <div class="mt-4 px-5 space-y-5">
          {#if $playgrounds.length > 0}
            <Table kind="playground" data="{$playgrounds}" columns="{columns}" row="{row}"></Table>
          {:else}
            <div role="status">
              There is no playground environment. You can <a
                href="{'javascript:void(0);'}"
                class="underline"
                role="button"
                title="Create a new Playground environment"
                on:click="{createNewPlayground}">create one now</a
              >.
            </div>
          {/if}
        </div>
      </div>
    </div>
  </svelte:fragment>
</NavPage>
