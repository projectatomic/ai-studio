<script lang="ts">
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import { router } from 'tinro';
export let object: ModelInfo;

function openDetails() {
  router.goto(`/model/${object.id}`);
}
</script>

<button class="flex flex-col w-full" title="{object.name}" on:click="{() => openDetails()}">
  <div class="text-sm text-gray-300 overflow-hidden text-ellipsis w-full">
    {object.name}
  </div>
  <span class="text-xs text-gray-700">{object.registry} - {object.license}</span>
</button>
