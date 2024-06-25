<script lang="ts">
import { tasks } from '/@/stores/tasks';
import type { Task } from '@shared/src/models/ITask';
import Card from '/@/lib/Card.svelte';
import TasksProgress from '/@/lib/progress/TasksProgress.svelte';
import { onMount } from 'svelte';

export let labels: string[] = [];
export let title: string;

let loadingTasks: Task[] = [];

onMount(() => {
  return tasks.subscribe(value => {
    loadingTasks = value.filter(
      task => task.state !== 'success' && labels.every(label => label in (task.labels ?? {})),
    );
  });
});
</script>

{#if loadingTasks.length > 0}
  <Card classes="bg-charcoal-800 mt-4 mx-5">
    <div slot="content" class="text-base font-normal p-2 w-full">
      <div class="text-base mb-2">{title}</div>
      <TasksProgress tasks="{loadingTasks}" />
    </div>
  </Card>
{/if}
