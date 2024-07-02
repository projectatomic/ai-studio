<script lang="ts">
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import ModelWhite from '../../icons/ModelWhite.svelte';
import { onMount } from 'svelte';
import { inferenceServers } from '/@/stores/inferenceServers';
import StatusIcon from '/@/lib/StatusIcon.svelte';
export let object: ModelInfo;

let status: string | undefined = undefined;
$: status;

onMount(() => {
  return inferenceServers.subscribe(servers => {
    if (servers.some(server => server.models.some(model => model.id === object.id))) {
      status = 'USED';
    } else {
      status = object.file ? 'DOWNLOADED' : 'NONE';
    }
  });
});
</script>

<StatusIcon status="{status}" icon="{ModelWhite}" />
