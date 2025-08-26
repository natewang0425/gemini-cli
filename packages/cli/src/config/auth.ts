/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { loadEnvironment } from './settings.js';
import * as readline from 'node:readline/promises';

export const validateAuthMethod = async (
  authMethod: string,
): Promise<string | null> => {
  loadEnvironment();
  if (
    authMethod === AuthType.LOGIN_WITH_GOOGLE ||
    authMethod === AuthType.CLOUD_SHELL
  ) {
    return null;
  }

  if (authMethod === AuthType.LOGIN_WITH_GOOGLE_GCA) {
    if (!process.env['GOOGLE_CLOUD_PROJECT']) {
      return (
        '[Error] GOOGLE_CLOUD_PROJECT is not set.\n' +
        'Please set it using:\n' +
        '  export GOOGLE_CLOUD_PROJECT=<your-project-id>\n' +
        'and try again.'
      );
    }
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    if (!process.env['GEMINI_API_KEY']) {
      return 'GEMINI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!';
    }
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env['GOOGLE_CLOUD_PROJECT'] &&
      !!process.env['GOOGLE_CLOUD_LOCATION'];
    const hasGoogleApiKey = !!process.env['GOOGLE_API_KEY'];
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return (
        'When using Vertex AI, you must specify either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your environment and try again (no reload needed if using .env)!'
      );
    }
    return null;
  }

  if (authMethod === AuthType.USE_OPENAI) {
    if (!process.env['OPENAI_API_KEY']) {
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });
      const apiKey = await rl.question('Please enter your OpenAI API key: ');
      rl.close();
      if (!apiKey) {
        return 'OPENAI_API_KEY not provided.';
      }
      process.env['OPENAI_API_KEY'] = apiKey;
    }
    return null;
  }

  return 'Invalid auth method selected.';
};
