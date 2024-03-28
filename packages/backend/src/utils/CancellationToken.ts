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
import {
  type CancellationToken,
  type CancellationTokenSource as ICancellationTokenSource,
  type Disposable,
  EventEmitter,
  type Event,
} from '@podman-desktop/api';

const shortcutEvent: Event<void> = Object.freeze(function (callback, context?): Disposable {
  const handle = setTimeout(callback.bind(context), 0);
  return {
    dispose(): void {
      clearTimeout(handle);
    },
  };
});

export class CancellationTokenImpl implements CancellationToken {
  private _isCancellationRequested: boolean;
  private emitter: EventEmitter<void> | undefined;

  constructor() {
    this._isCancellationRequested = false;
  }

  public cancel(): void {
    if (!this._isCancellationRequested) {
      this._isCancellationRequested = true;
      if (this.emitter) {
        this.emitter.fire(undefined);
        this.dispose();
      }
    }
  }

  get isCancellationRequested(): boolean {
    return this._isCancellationRequested;
  }

  get onCancellationRequested(): Event<void> {
    if (this._isCancellationRequested) {
      return shortcutEvent;
    }
    if (!this.emitter) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.emitter = new EventEmitter<void>();
    }
    return this.emitter.event;
  }

  public dispose(): void {
    if (this.emitter) {
      this.emitter.dispose();
      this.emitter = undefined;
    }
  }
}

export class CancellationTokenSource implements ICancellationTokenSource {
  private _token?: CancellationTokenImpl = undefined;

  get token(): CancellationToken {
    if (!this._token) {
      // be lazy and create the token only when actually needed
      this._token = new CancellationTokenImpl();
    }
    return this._token;
  }

  cancel(): void {
    if (this._token) {
      this._token?.cancel();
    }
  }

  dispose(cancel = false): void {
    if (cancel) {
      this.cancel();
    }
    if (this._token) {
      this._token.dispose();
    }
  }
}
