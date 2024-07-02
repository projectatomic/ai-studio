<script lang="ts">
import { inferenceServers } from '/@/stores/inferenceServers';
import ServiceStatus from '/@/lib/table/service/ServiceStatus.svelte';
import ServiceAction from '/@/lib/table/service/ServiceAction.svelte';
import Fa from 'svelte-fa';
import { faBuildingColumns, faCheck, faCopy, faMicrochip, faScaleBalanced } from '@fortawesome/free-solid-svg-icons';
import type { InferenceServer } from '@shared/src/models/IInference';
import { snippetLanguages } from '/@/stores/snippetLanguages';
import type { LanguageVariant } from 'postman-code-generators';
import { studioClient } from '/@/utils/client';
import { onMount } from 'svelte';
import { router } from 'tinro';
import Badge from '/@/lib/Badge.svelte';
import { Button, DetailsPage } from '@podman-desktop/ui-svelte';

export let containerId: string | undefined = undefined;

let service: InferenceServer | undefined = undefined;

let selectedLanguage: string = 'curl';
$: selectedLanguage;

let variants: LanguageVariant[] = [];
$: variants = $snippetLanguages.find(language => language.key === selectedLanguage)?.variants || [];

let selectedVariant: string = 'cURL';
$: selectedVariant;

const onLanguageChange = (): void => {
  if (variants.length > 0) {
    selectedVariant = variants[0].key;
    generate(selectedLanguage, selectedVariant);
  }
};

let snippet: string | undefined = undefined;
$: snippet;

const generate = async (language: string, variant: string) => {
  copied = false;
  snippet = await studioClient.createSnippet(
    {
      url: `http://localhost:${service?.connection.port || '??'}/v1/chat/completions`,
      method: 'POST',
      header: [
        {
          key: 'Content-Type',
          value: 'application/json',
        },
      ],
      body: {
        mode: 'raw',
        raw: `{
  "messages": [
    {
      "content": "You are a helpful assistant.",
      "role": "system"
    },
    {
      "content": "What is the capital of France?",
      "role": "user"
    }
  ]
}`,
      },
    },
    language,
    variant,
  );
};

$: {
  if (!snippet && service) {
    generate('curl', 'cURL');
  }
}

let code: HTMLElement;

$: {
  if (code) {
    code.addEventListener('copy', event => {
      studioClient.telemetryLogUsage('snippet.copy', {
        cpyButton: false,
        language: selectedLanguage,
        variant: selectedVariant,
      });
    });
  }
}

let copied: boolean = false;
function copySnippet(): void {
  if (!snippet) return;

  studioClient
    .copyToClipboard(snippet)
    .then(() => {
      copied = true;
      studioClient.telemetryLogUsage('snippet.copy', {
        cpyButton: true,
        language: selectedLanguage,
        variant: selectedVariant,
      });
    })
    .catch((err: unknown) => {
      console.error('Something went wrong while trying to copy language snippet.', err);
    });
}

onMount(() => {
  return inferenceServers.subscribe(servers => {
    service = servers.find(server => server.container.containerId === containerId);
    if (!service) {
      router.goto('/services');
    }
  });
});

export function goToUpPage(): void {
  router.goto('/services');
}
</script>

<DetailsPage
  title="Service details"
  breadcrumbLeftPart="Model Services"
  breadcrumbRightPart="Service details"
  breadcrumbTitle="Go back to Model Services"
  on:close="{goToUpPage}"
  on:breadcrumbClick="{goToUpPage}">
  <svelte:fragment slot="icon">
    <div class="mr-3">
      {#if service !== undefined}
        <ServiceStatus object="{service}" />
      {/if}
    </div>
  </svelte:fragment>
  <svelte:fragment slot="subtitle">
    <div class="flex gap-x-2 items-center text-[var(--pd-content-sub-header)]">
      {#if service}
        <span class="text-xs">{service.container.containerId}</span>
        {#each service.models as model}
          <Badge icon="{faMicrochip}" content="{model.hw}" background="bg-[var(--pd-label-bg)]" />
        {/each}
      {/if}
    </div>
  </svelte:fragment>
  <svelte:fragment slot="actions">
    {#if service !== undefined}
      <ServiceAction detailed object="{service}" />
    {/if}
  </svelte:fragment>
  <svelte:fragment slot="content">
    <div class="h-full overflow-y-auto bg-[var(--pd-content-bg)]">
      <div class="flex flex-col min-w-full min-h-full">
        <div class="min-w-full min-h-full flex-1">
          <div class="mt-4 px-5 space-y-5">
            {#if service !== undefined}
              <!-- models info -->
              <div class="bg-[var(--pd-content-card-bg)] rounded-md w-full px-4 pt-2 pb-4">
                <div>
                  <span class="text-sm text-[var(--pd-content-card-text)]">Models</span>
                  <div
                    class="w-full bg-[var(--pd-label-bg)] text-[var(--pd-label-text)] rounded-md p-2 flex flex-col gap-y-4">
                    {#each service.models as model}
                      <div class="flex flex-row gap-2 items-center">
                        <div class="grow text-sm" aria-label="Model name">
                          <a href="/model/{model.id}">{model.name}</a>
                        </div>
                        <div>
                          <div
                            class="bg-[var(--pd-content-card-bg)] rounded-md p-2 flex flex-row w-min h-min text-xs text-charcoal-100 text-nowrap items-center">
                            <Fa class="mr-2" icon="{faScaleBalanced}" />
                            {model.license}
                          </div>
                        </div>
                        <div>
                          <div
                            class="bg-[var(--pd-content-card-bg)] rounded-md p-2 flex flex-row w-min h-min text-xs text-charcoal-100 text-nowrap items-center">
                            <Fa class="mr-2" icon="{faBuildingColumns}" />
                            {model.registry}
                          </div>
                        </div>
                      </div>
                    {/each}
                  </div>
                </div>
              </div>

              <!-- server details -->
              <div class="bg-[var(--pd-content-card-bg)] rounded-md w-full px-4 pt-2 pb-4 mt-2">
                <span class="text-sm text-[var(--pd-content-card-text)]">Server</span>
                <div class="flex flex-row gap-4">
                  <div
                    class="bg-[var(--pd-label-bg)] text-[var(--pd-label-text)] rounded-md p-2 flex flex-row w-min h-min text-xs text-nowrap items-center">
                    http://localhost:{service.connection.port}/v1
                  </div>

                  <div
                    class="bg-[var(--pd-label-bg)] text-[var(--pd-label-text)] rounded-md p-2 flex flex-row w-min h-min text-xs text-nowrap items-center">
                    CPU Inference
                    <Fa class="ml-2" icon="{faMicrochip}" />
                  </div>
                </div>
              </div>

              <!-- code client -->
              <div>
                <div class="flex flex-row">
                  <span class="text-base grow text-[var(--pd-content-card-text)]">Client code</span>

                  <!-- language choice -->
                  <select
                    required
                    aria-label="snippet language selection"
                    bind:value="{selectedLanguage}"
                    on:change="{onLanguageChange}"
                    id="languages"
                    class="border ml-1 text-sm rounded-lg bg-[var(--pd-action-button-details-bg)] block p-1 border-[var(--pd-action-button-details-bg)] placeholder-gray-700 text-[var(--pd-action-button-details-text)]"
                    name="languages">
                    {#each $snippetLanguages as language}
                      <option class="my-1" value="{language.key}">{language.label}</option>
                    {/each}
                  </select>
                  {#if selectedVariant !== undefined}
                    <select
                      required
                      aria-label="snippet language variant"
                      id="variants"
                      bind:value="{selectedVariant}"
                      on:change="{() => generate(selectedLanguage, selectedVariant)}"
                      disabled="{variants.length === 1}"
                      class="border ml-1 text-sm rounded-lg bg-[var(--pd-action-button-details-bg)] block p-1 border-[var(--pd-action-button-details-bg)] placeholder-gray-700 text-[var(--pd-action-button-details-text)]"
                      name="variants">
                      {#each variants as variant}
                        <option class="my-1" value="{variant.key}">{variant.key}</option>
                      {/each}
                    </select>
                  {/if}
                </div>

                {#if snippet !== undefined}
                  <div
                    class="bg-[var(--pd-details-empty-cmdline-bg)] text-[var(--pd-details-empty-cmdline-text)] rounded-md w-full p-4 mt-2 relative">
                    <code class="whitespace-break-spaces text-sm" bind:this="{code}">
                      {snippet}
                    </code>
                    <div class="absolute right-4 top-4 z-10">
                      <Button
                        icon="{copied ? faCheck : faCopy}"
                        type="secondary"
                        title="Copy"
                        on:click="{copySnippet}" />
                    </div>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </svelte:fragment>
</DetailsPage>
