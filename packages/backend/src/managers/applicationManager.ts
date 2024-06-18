/**********************************************************************
 * Copyright (C) 2024 Red Hat, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ***********************************************************************/

import type { Recipe } from '@shared/src/models/IRecipe';
import type { GitCloneInfo, GitManager } from './gitManager';
import fs from 'fs';
import * as path from 'node:path';
import { containerEngine, Disposable, window, ProgressLocation } from '@podman-desktop/api';
import type {
  PodCreatePortOptions,
  TelemetryLogger,
  PodInfo,
  Webview,
  HostConfig,
  HealthConfig,
  PodContainerInfo,
} from '@podman-desktop/api';
import type { AIConfig, AIConfigFile, ContainerConfig } from '../models/AIConfig';
import { parseYamlFile } from '../models/AIConfig';
import type { Task } from '@shared/src/models/ITask';
import type { ModelInfo } from '@shared/src/models/IModelInfo';
import type { ModelsManager } from './modelsManager';
import { getPortsFromLabel, getPortsInfo } from '../utils/ports';
import { goarch } from '../utils/arch';
import { getDurationSecondsSince, timeout } from '../utils/utils';
import type { LocalRepositoryRegistry } from '../registries/LocalRepositoryRegistry';
import type { ApplicationState } from '@shared/src/models/IApplicationState';
import type { PodmanConnection } from './podmanConnection';
import { Messages } from '@shared/Messages';
import type { CatalogManager } from './catalogManager';
import { ApplicationRegistry } from '../registries/ApplicationRegistry';
import type { TaskRegistry } from '../registries/TaskRegistry';
import { Publisher } from '../utils/Publisher';
import { isQEMUMachine } from '../utils/podman';
import { getModelPropertiesForEnvironment } from '../utils/modelsUtils';
import { getRandomName, getRandomString } from '../utils/randomUtils';
import type { BuilderManager } from './recipes/BuilderManager';
import type { PodManager } from './recipes/PodManager';
import { SECOND } from '../workers/provider/LlamaCppPython';

export const LABEL_MODEL_ID = 'ai-lab-model-id';
export const LABEL_MODEL_PORTS = 'ai-lab-model-ports';

export const LABEL_RECIPE_ID = 'ai-lab-recipe-id';
export const LABEL_APP_PORTS = 'ai-lab-app-ports';

export const CONFIG_FILENAME = 'ai-lab.yaml';

interface AIContainers {
  aiConfigFile: AIConfigFile;
  containers: ContainerConfig[];
}

export interface ImageInfo {
  id: string;
  modelService: boolean;
  ports: string[];
  appName: string;
}

export class ApplicationManager extends Publisher<ApplicationState[]> implements Disposable {
  #applications: ApplicationRegistry<ApplicationState>;
  protectTasks: Set<string> = new Set();
  #disposables: Disposable[];

  constructor(
    private appUserDirectory: string,
    private git: GitManager,
    private taskRegistry: TaskRegistry,
    webview: Webview,
    private podmanConnection: PodmanConnection,
    private catalogManager: CatalogManager,
    private modelsManager: ModelsManager,
    private telemetry: TelemetryLogger,
    private localRepositories: LocalRepositoryRegistry,
    private builderManager: BuilderManager,
    private podManager: PodManager,
  ) {
    super(webview, Messages.MSG_APPLICATIONS_STATE_UPDATE, () => this.getApplicationsState());
    this.#applications = new ApplicationRegistry<ApplicationState>();
    this.#disposables = [];
  }

  async requestPullApplication(recipe: Recipe, model: ModelInfo): Promise<string> {
    // create a tracking id to put in the labels
    const trackingId: string = getRandomString();

    const labels: Record<string, string> = {
      trackingId: trackingId,
    };

    const task = this.taskRegistry.createTask(`Pulling ${recipe.name} recipe`, 'loading', labels);

    window
      .withProgress({ location: ProgressLocation.TASK_WIDGET, title: `Pulling ${recipe.name}.` }, () =>
        this.pullApplication(recipe, model, labels),
      )
      .then(() => {
        task.state = 'success';
      })
      .catch((err: unknown) => {
        task.state = 'error';
        task.error = `Something went wrong while pulling ${recipe.name}: ${String(err)}`;
      })
      .finally(() => {
        this.taskRegistry.updateTask(task);
      });

    return trackingId;
  }

  async pullApplication(recipe: Recipe, model: ModelInfo, labels: Record<string, string> = {}): Promise<void> {
    // clear any existing status / tasks related to the pair recipeId-modelId.
    this.taskRegistry.deleteByLabels({
      'recipe-id': recipe.id,
      'model-id': model.id,
    });

    const startTime = performance.now();
    try {
      // init application (git clone, models download etc.)
      const podInfo: PodInfo = await this.initApplication(recipe, model, labels);
      // start the pod
      await this.runApplication(podInfo, {
        ...labels,
        'recipe-id': recipe.id,
        'model-id': model.id,
      });

      // measure init + start time
      const durationSeconds = getDurationSecondsSince(startTime);
      this.telemetry.logUsage('recipe.pull', { 'recipe.id': recipe.id, 'recipe.name': recipe.name, durationSeconds });
    } catch (err: unknown) {
      const durationSeconds = getDurationSecondsSince(startTime);
      this.telemetry.logError('recipe.pull', {
        'recipe.id': recipe.id,
        'recipe.name': recipe.name,
        durationSeconds,
        message: 'error pulling application',
        error: err,
      });
      throw err;
    }
  }

  public async cloneApplication(recipe: Recipe, labels?: { [key: string]: string }): Promise<void> {
    const localFolder = path.join(this.appUserDirectory, recipe.id);

    // clone the recipe repository on the local folder
    const gitCloneInfo: GitCloneInfo = {
      repository: recipe.repository,
      ref: recipe.ref,
      targetDirectory: localFolder,
    };
    await this.doCheckout(gitCloneInfo, {
      ...labels,
      'recipe-id': recipe.id,
    });

    this.localRepositories.register({
      path: gitCloneInfo.targetDirectory,
      sourcePath: path.join(gitCloneInfo.targetDirectory, recipe.basedir ?? ''),
      labels: {
        'recipe-id': recipe.id,
      },
    });
  }

  /**
   * This method will execute the following tasks
   * - git clone
   * - git checkout
   * - register local repository
   * - download models
   * - upload models
   * - build containers
   * - create pod
   *
   * @param recipe
   * @param model
   * @param labels
   * @private
   */
  private async initApplication(
    recipe: Recipe,
    model: ModelInfo,
    labels: Record<string, string> = {},
  ): Promise<PodInfo> {
    const localFolder = path.join(this.appUserDirectory, recipe.id);

    // clone the application
    await this.cloneApplication(recipe, { ...labels, 'model-id': model.id });

    // load and parse the recipe configuration file and filter containers based on architecture, gpu accelerator
    // and backend (that define which model supports)
    const configAndFilteredContainers = this.getConfigAndFilterContainers(recipe.basedir, localFolder);

    // get model by downloading it or retrieving locally
    await this.modelsManager.requestDownloadModel(model, {
      ...labels,
      'recipe-id': recipe.id,
      'model-id': model.id,
    });

    // upload model to podman machine if user system is supported
    const modelPath = await this.modelsManager.uploadModelToPodmanMachine(model, {
      ...labels,
      'recipe-id': recipe.id,
      'model-id': model.id,
    });

    // build all images, one per container (for a basic sample we should have 2 containers = sample app + model service)
    const images = await this.builderManager.build(
      recipe,
      configAndFilteredContainers.containers,
      configAndFilteredContainers.aiConfigFile.path,
      {
        ...labels,
        'recipe-id': recipe.id,
        'model-id': model.id,
      },
    );

    // first delete any existing pod with matching labels
    if (await this.hasApplicationPod(recipe.id, model.id)) {
      await this.removeApplication(recipe.id, model.id);
    }

    // create a pod containing all the containers to run the application
    return this.createApplicationPod(recipe, model, images, modelPath, {
      ...labels,
      'recipe-id': recipe.id,
      'model-id': model.id,
    });
  }

  /**
   * Given an ApplicationPodInfo, start the corresponding pod
   * @param podInfo
   * @param labels
   */
  async runApplication(podInfo: PodInfo, labels?: { [key: string]: string }): Promise<void> {
    const task = this.taskRegistry.createTask('Starting AI App', 'loading', labels);

    // it starts the pod
    await this.podManager.startPod(podInfo.engineId, podInfo.Id);

    // check if all containers have started successfully
    for (const container of podInfo.Containers ?? []) {
      await this.waitContainerIsRunning(podInfo.engineId, container);
    }

    // Update task registry
    this.taskRegistry.updateTask({
      ...task,
      state: 'success',
      name: 'AI App is running',
    });

    return this.checkPodsHealth();
  }

  async waitContainerIsRunning(engineId: string, container: PodContainerInfo): Promise<void> {
    const TIME_FRAME_MS = 5000;
    const MAX_ATTEMPTS = 60 * (60000 / TIME_FRAME_MS); // try for 1 hour
    for (let i = 0; i < MAX_ATTEMPTS; i++) {
      const sampleAppContainerInspectInfo = await containerEngine.inspectContainer(engineId, container.Id);
      if (sampleAppContainerInspectInfo.State.Running) {
        return;
      }
      await timeout(TIME_FRAME_MS);
    }
    throw new Error(`Container ${container.Id} not started in time`);
  }

  async createApplicationPod(
    recipe: Recipe,
    model: ModelInfo,
    images: ImageInfo[],
    modelPath: string,
    labels?: { [key: string]: string },
  ): Promise<PodInfo> {
    const task = this.taskRegistry.createTask('Creating AI App', 'loading', labels);

    // create empty pod
    let podInfo: PodInfo;
    try {
      podInfo = await this.createPod(recipe, model, images);
      task.labels = {
        ...task.labels,
        'pod-id': podInfo.Id,
      };
    } catch (e) {
      console.error('error when creating pod', e);
      task.state = 'error';
      task.error = `Something went wrong while creating pod: ${String(e)}`;
      throw e;
    } finally {
      this.taskRegistry.updateTask(task);
    }

    try {
      await this.createContainerAndAttachToPod(podInfo, images, model, modelPath);
      task.state = 'success';
    } catch (e) {
      console.error(`error when creating pod ${podInfo.Id}`, e);
      task.state = 'error';
      task.error = `Something went wrong while creating pod: ${String(e)}`;
      throw e;
    } finally {
      this.taskRegistry.updateTask(task);
    }

    return podInfo;
  }

  async createContainerAndAttachToPod(
    podInfo: PodInfo,
    images: ImageInfo[],
    modelInfo: ModelInfo,
    modelPath: string,
  ): Promise<void> {
    // temporary check to set Z flag or not - to be removed when switching to podman 5
    const isQEMUVM = await isQEMUMachine();
    await Promise.all(
      images.map(async image => {
        let hostConfig: HostConfig | undefined = undefined;
        let envs: string[] = [];
        let healthcheck: HealthConfig | undefined = undefined;
        // if it's a model service we mount the model as a volume
        if (image.modelService) {
          const modelName = path.basename(modelPath);
          hostConfig = {
            Mounts: [
              {
                Target: `/${modelName}`,
                Source: modelPath,
                Type: 'bind',
                Mode: isQEMUVM ? undefined : 'Z',
              },
            ],
          };
          envs = [`MODEL_PATH=/${modelName}`];
          envs.push(...getModelPropertiesForEnvironment(modelInfo));
        } else {
          // TODO: remove static port
          const modelService = images.find(image => image.modelService);
          if (modelService && modelService.ports.length > 0) {
            const endPoint = `http://localhost:${modelService.ports[0]}`;
            envs = [`MODEL_ENDPOINT=${endPoint}`];
          }
        }
        if (image.ports.length > 0) {
          healthcheck = {
            // must be the port INSIDE the container not the exposed one
            Test: ['CMD-SHELL', `curl -s localhost:${image.ports[0]} > /dev/null`],
            Interval: SECOND * 5,
            Retries: 4 * 5,
            Timeout: SECOND * 2,
          };
        }

        const podifiedName = getRandomName(`${image.appName}-podified`);
        await containerEngine.createContainer(podInfo.engineId, {
          Image: image.id,
          name: podifiedName,
          Detach: true,
          HostConfig: hostConfig,
          Env: envs,
          start: false,
          pod: podInfo.Id,
          HealthCheck: healthcheck,
        });
      }),
    );
  }

  async createPod(recipe: Recipe, model: ModelInfo, images: ImageInfo[]): Promise<PodInfo> {
    // find the exposed port of the sample app so we can open its ports on the new pod
    const sampleAppImageInfo = images.find(image => !image.modelService);
    if (!sampleAppImageInfo) {
      console.error('no sample app image found');
      throw new Error('no sample app found');
    }

    const portmappings: PodCreatePortOptions[] = [];
    // we expose all ports so we can check the model service if it is actually running
    for (const image of images) {
      for (const exposed of image.ports) {
        const localPorts = await getPortsInfo(exposed);
        if (localPorts) {
          portmappings.push({
            container_port: parseInt(exposed),
            host_port: parseInt(localPorts),
            host_ip: '',
            protocol: '',
            range: 1,
          });
        }
      }
    }

    // create new pod
    const labels: Record<string, string> = {
      [LABEL_RECIPE_ID]: recipe.id,
      [LABEL_MODEL_ID]: model.id,
    };
    const modelPorts = images
      .filter(img => img.modelService)
      .flatMap(img => img.ports)
      .map(port => portmappings.find(pm => `${pm.container_port}` === port)?.host_port);
    if (modelPorts.length) {
      labels[LABEL_MODEL_PORTS] = modelPorts.join(',');
    }
    const appPorts = images
      .filter(img => !img.modelService)
      .flatMap(img => img.ports)
      .map(port => portmappings.find(pm => `${pm.container_port}` === port)?.host_port);
    if (appPorts.length) {
      labels[LABEL_APP_PORTS] = appPorts.join(',');
    }
    const { engineId, Id } = await this.podManager.createPod({
      name: getRandomName(`pod-${sampleAppImageInfo.appName}`),
      portmappings: portmappings,
      labels,
    });

    return this.podManager.getPod(engineId, Id);
  }

  /**
   * Stop the pod with matching recipeId and modelId
   * @param recipeId
   * @param modelId
   */
  async stopApplication(recipeId: string, modelId: string): Promise<PodInfo> {
    // clear existing tasks
    this.clearTasks(recipeId, modelId);

    // get the application pod
    const appPod = await this.getApplicationPod(recipeId, modelId);

    // if the pod is already stopped skip
    if (appPod.Status === 'Exited') {
      return appPod;
    }

    // create a task to follow progress/error
    const stoppingTask = this.taskRegistry.createTask(`Stopping AI App`, 'loading', {
      'recipe-id': recipeId,
      'model-id': modelId,
    });

    try {
      await this.podManager.stopPod(appPod.engineId, appPod.Id);

      stoppingTask.state = 'success';
      stoppingTask.name = `AI App Stopped`;
    } catch (err: unknown) {
      stoppingTask.error = `Error removing the pod.: ${String(err)}`;
      stoppingTask.name = 'Error stopping AI App';
    } finally {
      this.taskRegistry.updateTask(stoppingTask);
      await this.checkPodsHealth();
    }
    return appPod;
  }

  /**
   * Utility method to start a pod using (recipeId, modelId)
   * @param recipeId
   * @param modelId
   */
  async startApplication(recipeId: string, modelId: string): Promise<void> {
    this.clearTasks(recipeId, modelId);
    const pod = await this.getApplicationPod(recipeId, modelId);

    return this.runApplication(pod, {
      'recipe-id': recipeId,
      'model-id': modelId,
    });
  }

  private getConfigAndFilterContainers(
    recipeBaseDir: string | undefined,
    localFolder: string,
    labels?: { [key: string]: string },
  ): AIContainers {
    // Adding loading configuration task
    const task = this.taskRegistry.createTask('Loading configuration', 'loading', labels);

    let aiConfigFile: AIConfigFile;
    try {
      // load and parse the recipe configuration file
      aiConfigFile = this.getConfiguration(recipeBaseDir, localFolder);
    } catch (e) {
      task.error = `Something went wrong while loading configuration: ${String(e)}.`;
      this.taskRegistry.updateTask(task);
      throw e;
    }

    // filter the containers based on architecture, gpu accelerator and backend (that define which model supports)
    const filteredContainers: ContainerConfig[] = this.filterContainers(aiConfigFile.aiConfig);
    if (filteredContainers.length > 0) {
      // Mark as success.
      task.state = 'success';
      this.taskRegistry.updateTask(task);
    } else {
      // Mark as failure.
      task.error = 'No containers available.';
      this.taskRegistry.updateTask(task);
      throw new Error('No containers available.');
    }

    return {
      aiConfigFile: aiConfigFile,
      containers: filteredContainers,
    };
  }

  filterContainers(aiConfig: AIConfig): ContainerConfig[] {
    return aiConfig.application.containers.filter(
      container => container.gpu_env.length === 0 && container.arch.some(arc => arc === goarch()),
    );
  }

  getConfiguration(recipeBaseDir: string | undefined, localFolder: string): AIConfigFile {
    let configFile: string;
    if (recipeBaseDir !== undefined) {
      configFile = path.join(localFolder, recipeBaseDir, CONFIG_FILENAME);
    } else {
      configFile = path.join(localFolder, CONFIG_FILENAME);
    }

    if (!fs.existsSync(configFile)) {
      throw new Error(`The file located at ${configFile} does not exist.`);
    }

    // If the user configured the config as a directory we check for "ai-lab.yaml" inside.
    if (fs.statSync(configFile).isDirectory()) {
      const tmpPath = path.join(configFile, CONFIG_FILENAME);
      // If it has the ai-lab.yaml we use it.
      if (fs.existsSync(tmpPath)) {
        configFile = tmpPath;
      }
    }

    // Parsing the configuration
    console.log(`Reading configuration from ${configFile}.`);
    let aiConfig: AIConfig;
    try {
      aiConfig = parseYamlFile(configFile, goarch());
    } catch (err) {
      console.error('Cannot load configure file.', err);
      throw new Error(`Cannot load configuration file.`);
    }

    // Mark as success.
    return {
      aiConfig,
      path: configFile,
    };
  }

  private async doCheckout(gitCloneInfo: GitCloneInfo, labels?: { [id: string]: string }): Promise<void> {
    // Creating checkout task
    const checkoutTask: Task = this.taskRegistry.createTask('Checking out repository', 'loading', {
      ...labels,
      git: 'checkout',
    });

    try {
      await this.git.processCheckout(gitCloneInfo);
      checkoutTask.state = 'success';
    } catch (err: unknown) {
      checkoutTask.state = 'error';
      checkoutTask.error = String(err);
      // propagate error
      throw err;
    } finally {
      // Update task registry
      this.taskRegistry.updateTask(checkoutTask);
    }
  }

  init() {
    this.podmanConnection.startupSubscribe(() => {
      this.podManager
        .getPodsWithLabels([LABEL_RECIPE_ID])
        .then(pods => {
          pods.forEach(pod => this.adoptPod(pod));
        })
        .catch((err: unknown) => {
          console.error('error during adoption of existing playground containers', err);
        });
    });

    this.podmanConnection.onMachineStop(() => {
      // Podman Machine has been stopped, we consider all recipe pods are stopped
      for (const recipeModelIndex of this.#applications.keys()) {
        this.taskRegistry.createTask('AI App stopped manually', 'success', {
          'recipe-id': recipeModelIndex.recipeId,
          'model-id': recipeModelIndex.modelId,
        });
      }

      this.#applications.clear();
      this.notify();
    });

    this.podManager.onStartPodEvent((pod: PodInfo) => {
      this.adoptPod(pod);
    });
    this.podManager.onRemovePodEvent(({ podId }) => {
      this.forgetPodById(podId);
    });

    const ticker = () => {
      this.checkPodsHealth()
        .catch((err: unknown) => {
          console.error('error getting pods statuses', err);
        })
        .finally(() => (timerId = setTimeout(ticker, 10000)));
    };

    // using a recursive setTimeout instead of setInterval as we don't know how long the operation takes
    let timerId = setTimeout(ticker, 1000);

    this.#disposables.push(
      Disposable.create(() => {
        clearTimeout(timerId);
      }),
    );
  }

  private adoptPod(pod: PodInfo) {
    if (!pod.Labels) {
      return;
    }
    const recipeId = pod.Labels[LABEL_RECIPE_ID];
    const modelId = pod.Labels[LABEL_MODEL_ID];
    if (!recipeId || !modelId) {
      return;
    }
    const appPorts = getPortsFromLabel(pod.Labels, LABEL_APP_PORTS);
    const modelPorts = getPortsFromLabel(pod.Labels, LABEL_MODEL_PORTS);
    if (this.#applications.has({ recipeId, modelId })) {
      return;
    }
    const state: ApplicationState = {
      recipeId,
      modelId,
      pod,
      appPorts,
      modelPorts,
      health: 'starting',
    };
    this.updateApplicationState(recipeId, modelId, state);
  }

  private forgetPodById(podId: string) {
    const app = Array.from(this.#applications.values()).find(p => p.pod.Id === podId);
    if (!app) {
      return;
    }
    if (!app.pod.Labels) {
      return;
    }
    const recipeId = app.pod.Labels[LABEL_RECIPE_ID];
    const modelId = app.pod.Labels[LABEL_MODEL_ID];
    if (!recipeId || !modelId) {
      return;
    }
    if (!this.#applications.has({ recipeId, modelId })) {
      return;
    }
    this.#applications.delete({ recipeId, modelId });
    this.notify();

    const protect = this.protectTasks.has(podId);
    if (!protect) {
      this.taskRegistry.createTask('AI App stopped manually', 'success', {
        'recipe-id': recipeId,
        'model-id': modelId,
      });
    } else {
      this.protectTasks.delete(podId);
    }
  }

  private async checkPodsHealth(): Promise<void> {
    const pods = await this.podManager.getPodsWithLabels([LABEL_RECIPE_ID, LABEL_MODEL_ID]);
    let changes = false;

    for (const pod of pods) {
      const recipeId = pod.Labels[LABEL_RECIPE_ID];
      const modelId = pod.Labels[LABEL_MODEL_ID];
      if (!this.#applications.has({ recipeId, modelId })) {
        // a fresh pod could not have been added yet, we will handle it at next iteration
        continue;
      }

      const podHealth = await this.podManager.getHealth(pod);
      const state = this.#applications.get({ recipeId, modelId });
      if (state.health !== podHealth) {
        state.health = podHealth;
        state.pod = pod;
        this.#applications.set({ recipeId, modelId }, state);
        changes = true;
      }
      if (pod.Status !== state.pod.Status) {
        state.pod = pod;
        changes = true;
      }
    }
    if (changes) {
      this.notify();
    }
  }

  updateApplicationState(recipeId: string, modelId: string, state: ApplicationState): void {
    this.#applications.set({ recipeId, modelId }, state);
    this.notify();
  }

  getApplicationsState(): ApplicationState[] {
    return Array.from(this.#applications.values());
  }

  private clearTasks(recipeId: string, modelId: string): void {
    // clear any existing status / tasks related to the pair recipeId-modelId.
    this.taskRegistry.deleteByLabels({
      'recipe-id': recipeId,
      'model-id': modelId,
    });
  }

  /**
   * Method that will stop then remove a pod corresponding to the recipe and model provided
   * @param recipeId
   * @param modelId
   */
  async removeApplication(recipeId: string, modelId: string): Promise<void> {
    const appPod = await this.stopApplication(recipeId, modelId);

    const remoteTask = this.taskRegistry.createTask(`Removing AI App`, 'loading', {
      'recipe-id': recipeId,
      'model-id': modelId,
    });
    // protect the task
    this.protectTasks.add(appPod.Id);

    try {
      await this.podManager.removePod(appPod.engineId, appPod.Id);

      remoteTask.state = 'success';
      remoteTask.name = `AI App Removed`;
    } catch (err: unknown) {
      remoteTask.error = 'error removing the pod. Please try to remove the pod manually';
      remoteTask.name = 'Error stopping AI App';
    } finally {
      this.taskRegistry.updateTask(remoteTask);
    }
  }

  async restartApplication(recipeId: string, modelId: string): Promise<void> {
    const appPod = await this.getApplicationPod(recipeId, modelId);
    await this.removeApplication(recipeId, modelId);
    const recipe = this.catalogManager.getRecipeById(recipeId);
    const model = this.catalogManager.getModelById(appPod.Labels[LABEL_MODEL_ID]);

    // init the recipe
    const podInfo = await this.initApplication(recipe, model);

    // start the pod
    return this.runApplication(podInfo, {
      'recipe-id': recipe.id,
      'model-id': model.id,
    });
  }

  async getApplicationPorts(recipeId: string, modelId: string): Promise<number[]> {
    const recipe = this.catalogManager.getRecipeById(recipeId);
    const state = this.#applications.get({ recipeId, modelId });
    if (state) {
      return state.appPorts;
    }
    throw new Error(`Recipe ${recipe.name} has no ports available`);
  }

  async getApplicationPod(recipeId: string, modelId: string): Promise<PodInfo> {
    const appPod = await this.findPod(recipeId, modelId);
    if (!appPod) {
      throw new Error(`no pod found with recipe Id ${recipeId} and model Id ${modelId}`);
    }
    return appPod;
  }

  private async hasApplicationPod(recipeId: string, modelId: string): Promise<boolean> {
    const pod = await this.podManager.findPodByLabelsValues({
      LABEL_RECIPE_ID: recipeId,
      LABEL_MODEL_ID: modelId,
    });
    return !!pod;
  }

  private async findPod(recipeId: string, modelId: string): Promise<PodInfo | undefined> {
    return this.podManager.findPodByLabelsValues({
      [LABEL_RECIPE_ID]: recipeId,
      [LABEL_MODEL_ID]: modelId,
    });
  }

  dispose(): void {
    this.#disposables.forEach(disposable => disposable.dispose());
  }
}
