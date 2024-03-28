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
import type { Disposable, Webview } from '@podman-desktop/api';
import { getLanguageList, convert, type Language } from 'postman-code-generators';
import { Request } from 'postman-collection';
import { Publisher } from '../utils/Publisher';
import { Messages } from '@shared/Messages';
import type { RequestOptions } from '@shared/src/models/RequestOptions';
import { quarkusLangchain4Jgenerator } from './snippets/quarkus-snippet';
import { javaOkHttpGenerator } from './snippets/java-okhttp-snippet';
import { pythonLangChainGenerator } from './snippets/python-langchain-snippet';

type Generator = (requestOptions: RequestOptions) => Promise<string>;

export class SnippetManager extends Publisher<Language[]> implements Disposable {
  #languages: Language[];
  #additionalGenerators: Map<string, Generator>;

  constructor(webview: Webview) {
    super(webview, Messages.MSG_SUPPORTED_LANGUAGES_UPDATE, () => this.getLanguageList());
  }

  addVariant(key: string, variant: string, generator: Generator): void {
    const original = this.#languages;
    const language = original.find((lang: Language) => lang.key === key);
    if (language) {
      if (!language.variants.find(v => v.key === variant)) {
        language.variants.push({ key: variant });
      }
      this.#additionalGenerators.set(`${key}/${variant}`, generator);
    }
  }

  getLanguageList(): Language[] {
    return this.#languages;
  }

  async generate(requestOptions: RequestOptions, language: string, variant: string): Promise<string> {
    const generator = this.#additionalGenerators.get(`${language}/${variant}`);
    if (generator) {
      return generator(requestOptions);
    }

    return new Promise((resolve, reject) => {
      const request = new Request(requestOptions);
      convert(language, variant, request, {}, (error: unknown, snippet: string) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(snippet);
      });
    });
  }

  init() {
    this.#languages = getLanguageList();
    this.#additionalGenerators = new Map<string, Generator>();
    this.addVariant('java', 'Quarkus Langchain4J', quarkusLangchain4Jgenerator);
    this.addVariant('java', 'OkHttp', javaOkHttpGenerator);
    this.addVariant('python', 'Python LangChain', pythonLangChainGenerator);
    // Notify the publisher
    this.notify();
  }

  dispose(): void {}
}
