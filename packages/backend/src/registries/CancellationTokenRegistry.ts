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
import { CancellationTokenSource, type Disposable } from '@podman-desktop/api';

export class CancellationTokenRegistry implements Disposable {
  #callbackId: number;
  #callbacksCancellableToken: Map<number, CancellationTokenSource>;

  constructor() {
    this.#callbackId = 0;
    this.#callbacksCancellableToken = new Map<number, CancellationTokenSource>();
  }

  createCancellationTokenSource(): number {
    // keep track of this request
    this.#callbackId++;

    const token = new CancellationTokenSource();

    // store the callback that will resolve the promise
    this.#callbacksCancellableToken.set(this.#callbackId, token);

    return this.#callbackId;
  }

  getCancellationTokenSource(id: number): CancellationTokenSource | undefined {
    if (this.hasCancellationTokenSource(id)) {
      return this.#callbacksCancellableToken.get(id);
    }
    return undefined;
  }

  hasCancellationTokenSource(id: number): boolean {
    return this.#callbacksCancellableToken.has(id);
  }

  cancel(tokenId: number): void {
    if (!this.hasCancellationTokenSource(tokenId))
      throw new Error(`Cancellation token with id ${tokenId} does not exist.`);
    this.getCancellationTokenSource(tokenId).cancel();
    this.#callbacksCancellableToken.delete(tokenId);
  }

  dispose(): void {
    Array.from(this.#callbacksCancellableToken.values()).forEach(source => {
      source.cancel();
      source.dispose();
    });
    this.#callbacksCancellableToken.clear();
  }
}
