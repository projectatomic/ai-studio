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

import { env, window } from '@podman-desktop/api';
import fs, { existsSync, mkdirSync, type NoParamCallback, rmSync, statSync } from 'node:fs';
import git, { type CallbackFsClient } from 'isomorphic-git';
import http from 'isomorphic-git/http/node';

export interface GitCloneInfo {
  repository: string;
  ref?: string;
  targetDirectory: string;
}

function getFs(): CallbackFsClient {
  if (env.isWindows) {
    return {
      chmod: fs.chmod,
      lstat: fs.lstat,
      mkdir: fs.mkdir,
      readFile: fs.readFile,
      readdir: fs.readdir,
      readlink: fs.readlink,
      rmdir: fs.rmdir,
      stat: fs.stat,
      symlink: (
        target: fs.PathLike,
        path: fs.PathLike,
        type_: fs.symlink.Type | undefined,
        callback: fs.NoParamCallback,
      ) => {
        if (!callback) {
          callback = type_ as unknown as NoParamCallback;
          type_ = undefined;
        }
        fs.symlink(target, path, type_, err => {
          if (err) {
            fs.writeFile(path, target.toString(), { mode: 0o0444 }, err => {
              callback(err);
            });
          } else {
            // eslint-disable-next-line no-null/no-null
            callback(null);
          }
        });
      },
      unlink: fs.unlink,
      writeFile: fs.writeFile,
    };
  }
  return fs;
}

export class GitManager {
  async cloneRepository(gitCloneInfo: GitCloneInfo) {
    // clone repo
    await git.clone({
      fs: getFs(),
      http,
      dir: gitCloneInfo.targetDirectory,
      url: gitCloneInfo.repository,
      ref: gitCloneInfo.ref,
      singleBranch: true,
      depth: 1,
    });
  }

  async getRepositoryRemotes(directory: string): Promise<
    {
      remote: string;
      url: string;
    }[]
  > {
    return git.listRemotes({ fs, dir: directory });
  }

  /* see https://isomorphic-git.org/docs/en/statusMatrix
   *
   * - The HEAD status is either absent (0) or present (1).
   * - The WORKDIR status is either absent (0), identical to HEAD (1), or different from HEAD (2).
   * - The STAGE status is either absent (0), identical to HEAD (1), identical to WORKDIR (2), or different from WORKDIR (3).
   *
   * // example StatusMatrix
   * [
   *   ["a.txt", 0, 2, 0], // new, untracked
   *   ["b.txt", 0, 2, 2], // added, staged
   *   ["c.txt", 0, 2, 3], // added, staged, with unstaged changes
   *   ["d.txt", 1, 1, 1], // unmodified
   *   ["e.txt", 1, 2, 1], // modified, unstaged
   *   ["f.txt", 1, 2, 2], // modified, staged
   *   ["g.txt", 1, 2, 3], // modified, staged, with unstaged changes
   *   ["h.txt", 1, 0, 1], // deleted, unstaged
   *   ["i.txt", 1, 0, 0], // deleted, staged
   *   ["j.txt", 1, 2, 0], // deleted, staged, with unstaged-modified changes (new file of the same name)
   *   ["k.txt", 1, 1, 0], // deleted, staged, with unstaged changes (new file of the same name)
   * ]
   */
  async getRepositoryStatus(directory: string): Promise<{
    modified: string[];
    created: string[];
    deleted: string[];
    clean: boolean;
  }> {
    const status = await git.statusMatrix({
      fs: getFs(),
      dir: directory,
    });

    const FILE = 0,
      HEAD = 1,
      WORKDIR = 2,
      STAGE = 3;

    const created = status.filter(row => row[HEAD] === 0 && row[WORKDIR] === 2).map(row => row[FILE]);

    const deleted = status
      .filter(row => row[HEAD] === 1 && (row[WORKDIR] === 0 || row[STAGE] === 0))
      .map(row => row[FILE]);

    const modified = status.filter(row => row[HEAD] === 1 && row[WORKDIR] === 2).map(row => row[FILE]);

    const notClean = status.filter(row => row[HEAD] !== 1 || row[WORKDIR] !== 1 || row[STAGE] !== 1);

    return {
      modified,
      created,
      deleted,
      clean: notClean.length === 0,
    };
  }

  async getCurrentCommit(directory: string): Promise<string> {
    return git.resolveRef({ fs: getFs(), dir: directory, ref: 'HEAD' });
  }

  async pull(directory: string): Promise<void> {
    return git.pull({
      fs: getFs(),
      http,
      dir: directory,
    });
  }

  async processCheckout(gitCloneInfo: GitCloneInfo): Promise<void> {
    // Check for existing cloned repository
    if (existsSync(gitCloneInfo.targetDirectory) && statSync(gitCloneInfo.targetDirectory).isDirectory()) {
      const result = await this.isRepositoryUpToDate(
        gitCloneInfo.targetDirectory,
        gitCloneInfo.repository,
        gitCloneInfo.ref,
      );

      if (result.ok) {
        return;
      }

      const error = `The repository "${gitCloneInfo.repository}" appears to have already been cloned and does not match the expected configuration: ${result.error}`;

      // Ask user
      const selected = await window.showWarningMessage(
        `${error} By continuing, the AI application may not run as expected. `,
        'Cancel',
        'Continue',
        result.updatable ? 'Update' : 'Reset',
      );

      switch (selected) {
        case undefined:
        case 'Cancel':
          throw new Error('Cancelled');
        case 'Continue':
          return;
        case 'Update':
          await this.pull(gitCloneInfo.targetDirectory);
          return;
        case 'Reset':
          rmSync(gitCloneInfo.targetDirectory, { recursive: true });
          break;
      }
    }

    // Create folder
    mkdirSync(gitCloneInfo.targetDirectory, { recursive: true });

    // Clone the repository
    console.log(`Cloning repository ${gitCloneInfo.repository} in ${gitCloneInfo.targetDirectory}.`);
    await this.cloneRepository(gitCloneInfo);
  }

  async isRepositoryUpToDate(
    directory: string,
    origin: string,
    ref?: string,
  ): Promise<{ ok?: boolean; updatable?: boolean; error?: string }> {
    // fetch updates
    await git.fetch({
      fs: getFs(),
      http,
      dir: directory,
    });

    const remotes = await this.getRepositoryRemotes(directory);

    if (!remotes.some(remote => remote.url === origin)) {
      return {
        error: `The local repository does not have remote ${origin} configured. Remotes: ${remotes
          .map(remote => `${remote.remote} ${remote.url} (fetch)`)
          .join(',')}`,
      };
    }

    const branch = await git.currentBranch({
      fs: getFs(),
      dir: directory,
    });

    if (!branch) {
      // when the repository is detached
      if (ref === undefined) {
        return { error: 'The local repository is detached.' };
      } else {
        const tag = await this.getTagCommitId(directory, ref);
        if (tag) {
          ref = tag;
        }
        const commit = await this.getCurrentCommit(directory);
        if (!commit.startsWith(ref)) {
          return { error: `The local repository is detached. HEAD is ${commit} expected ${ref}.` };
        }
      }
    }

    if (branch) {
      const tracking = await this.getTrackingBranch(directory, branch);
      if (ref && tracking !== `origin/${ref}`) {
        return {
          error: `The local repository is not tracking the right branch. (tracking ${tracking} when expected ${ref})`,
        };
      }

      const { behind, ahead } = await this.getBehindAhead(directory, branch);

      if (ahead !== 0) {
        return { error: `The local repository has ${ahead} commit(s) ahead.` };
      }
      if (behind !== 0) {
        return { ok: true, updatable: true };
      }
    }

    const status = await this.getRepositoryStatus(directory);
    if (status.modified.length > 0) {
      return { error: 'The local repository has modified files.' };
    } else if (status.created.length > 0) {
      return { error: 'The local repository has created files.' };
    } else if (status.deleted.length > 0) {
      return { error: 'The local repository has deleted files.' };
    } else if (!status.clean) {
      return { error: 'The local repository is not clean.' };
    }

    return { ok: true }; // If none of the error conditions are met
  }

  async getTrackingBranch(directory: string, branch: string): Promise<string | undefined> {
    const mergeRef = await git.getConfig({
      fs: getFs(),
      dir: directory,
      path: `branch.${branch}.merge`,
    });
    const remote = await git.getConfig({
      fs: getFs(),
      dir: directory,
      path: `branch.${branch}.remote`,
    });
    return mergeRef && remote ? `${remote}/${mergeRef.replace(/^refs\/heads\//, '')}` : undefined;
  }

  async getBehindAhead(dir: string, localBranch: string): Promise<{ behind: number; ahead: number }> {
    const remoteBranch = await this.getTrackingBranch(dir, localBranch);

    const remoteCommits = (
      await git.log({
        fs: getFs(),
        dir,
        ref: remoteBranch,
      })
    )
      .map(c => c.oid)
      .sort();
    const localCommits = (
      await git.log({
        fs: getFs(),
        dir,
        ref: localBranch,
      })
    )
      .map(c => c.oid)
      .sort();

    let behind = 0;
    let ahead = 0;
    while (remoteCommits.length && localCommits.length) {
      const remote = remoteCommits.pop();
      const local = localCommits.pop();
      if (!remote || !local) {
        break;
      }
      if (remote === local) {
        continue;
      }
      if (remote > local) {
        behind++;
        localCommits.push(local);
      } else {
        ahead++;
        remoteCommits.push(remote);
      }
    }
    return {
      behind: behind + remoteCommits.length,
      ahead: ahead + localCommits.length,
    };
  }

  async getTagCommitId(directory: string, tagName: string): Promise<string | undefined> {
    try {
      return await git.resolveRef({
        fs: getFs(),
        dir: directory,
        ref: tagName,
      });
    } catch {
      return undefined;
    }
  }
}
