<script lang="ts">
import { router } from 'tinro';
import type { PlaygroundV2 } from '@shared/src/models/IPlaygroundV2';
export let object: PlaygroundV2;

function openDetails() {
  router.goto(`/playground/${object.id}`);
}
</script>

<button on:click="{() => openDetails()}">
  <div class="text-sm text-gray-300 overflow-hidden text-ellipsis">
    {object.name}
  </div>
</button>
