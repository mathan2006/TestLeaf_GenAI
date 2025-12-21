import { Router, Request, Response, NextFunction } from "express";
import { callLLM } from "../services/llmClient.js";
import { evalWithMetric, evalWithFields, evalFaithfulness } from "../services/evalClient.js";
import { retrieveContext } from "../services/ragService.js";
import { ENV } from "../config/env.js";

const router = Router();

/**
 * Error handler middleware for async routes
 */
const asyncHandler =
  (fn: (req: Request, res: Response) => Promise<any>) =>
  (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res)).catch(next);
  };

/**
 * POST /api/llm/eval
 * LLM-only evaluation endpoint
 *
 * Request body:
 * {
 *   prompt: string (required),
 *   model?: string (optional, defaults to llama-3.3-70b-versatile),
 *   temperature?: number (optional, defaults to 0.7),
 *   metric?: string (optional, defaults to 'answer_relevancy')
 * }
 *
 * Response:
 * {
 *   prompt: string,
 *   model: string,
 *   provider: string,
 *   llmResponse: string,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/llm/eval",
  asyncHandler(async (req: Request, res: Response) => {
    const { prompt, model, temperature, metric } = req.body;

    // Validation
    if (!prompt) {
      return res.status(400).json({
        error: "Missing required field: prompt"
      });
    }

    // Validate temperature if provided
    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({
        error: "Temperature must be a number between 0 and 2"
      });
    }

    // Determine effective parameters
    const effectiveModel = model || "llama-3.3-70b-versatile";
    const effectiveTemperature = temperature !== undefined ? temperature : 0.7;
    const effectiveMetric = metric || "answer_relevancy"; // Changed default from faithfulness

    // Call LLM
    const llmResponse = await callLLM(prompt, effectiveModel, effectiveTemperature);
    console.log("LLM Response:", llmResponse);

    // Determine provider based on model
    const provider = effectiveModel.startsWith("llama-") || effectiveModel.startsWith("mixtral-") || 
                     effectiveModel.startsWith("gemma") || effectiveModel.startsWith("qwen") ? "groq" : "openai";

    // Evaluate with DeepEval using specified metric
    // For LLM-only (no RAG), answer_relevancy makes most sense
    // query = prompt, output = llmResponse
    const evalResult = await evalWithFields({
      query: prompt,
      output: llmResponse,
      metric: effectiveMetric,
      provider
    });
    console.log("Evaluation Result:", evalResult);

    // Use legacy fields for backward compatibility (populated from first successful result)
    res.json({
      prompt,
      model: effectiveModel,
      temperature: effectiveTemperature,
      provider,
      llmResponse,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation,
        // Include results array if available for multi-metric support
        ...(evalResult.results && { results: evalResult.results })
      }
    });
  })
);

/**
 * POST /api/rag/eval
 * RAG + LLM evaluation endpoint
 *
 * Request body:
 * {
 *   query: string (required),
 *   model?: string (optional, defaults to llama-3.3-70b-versatile),
 *   temperature?: number (optional, defaults to 0.7),
 *   metric?: string (optional, defaults to 'faithfulness')
 * }
 *
 * Response:
 * {
 *   query: string,
 *   context: string,
 *   prompt: string,
 *   llmResponse: string,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/rag/eval",
  asyncHandler(async (req: Request, res: Response) => {
    const { query, model, temperature, metric } = req.body;

    // Validation
    if (!query) {
      return res.status(400).json({
        error: "Missing required field: query"
      });
    }

    // Validate temperature if provided
    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({
        error: "Temperature must be a number between 0 and 2"
      });
    }

    // Determine effective parameters
    const effectiveModel = model || "llama-3.3-70b-versatile";
    const effectiveTemperature = temperature !== undefined ? temperature : 0.7;
    const effectiveMetric = metric || "faithfulness";

    // 1. Retrieve context from RAG
    const contextStr = await retrieveContext(query);

    // 2. Build RAG prompt
    const ragPrompt = `You are a helpful QA assistant. Using ONLY the following context, answer the question as accurately as possible. If the context does not contain the answer, say "I don't have enough information to answer that."

CONTEXT:
${contextStr}

QUESTION:
${query}

ANSWER:`;

    // 3. Call LLM with RAG prompt
    const llmResponse = await callLLM(ragPrompt, effectiveModel, effectiveTemperature);

    // Determine provider based on model
    const provider = effectiveModel.startsWith("llama-") || effectiveModel.startsWith("mixtral-") || 
                     effectiveModel.startsWith("gemma") || effectiveModel.startsWith("qwen") ? "groq" : "openai";

    // 4. Evaluate using specified metric
    // For RAG, we have context (as array) and output
    const evalResult = await evalWithFields({
      context: [contextStr], // Convert string to array
      output: llmResponse,
      metric: effectiveMetric,
      provider
    });

    res.json({
      query,
      context: contextStr,
      prompt: ragPrompt,
      model: effectiveModel,
      temperature: effectiveTemperature,
      provider,
      llmResponse,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation,
        // Include results array if available for multi-metric support
        ...(evalResult.results && { results: evalResult.results })
      }
    });
  })
);

/**
 * GET /health
 * Health check endpoint
 */
router.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString()
  });
});

/**
 * POST /eval-only
 * Evaluate existing query-output pairs without LLM generation
 * 
 * Request body:
 * {
 *   query?: string - the input question (required for answer_relevancy),
 *   output?: string - the response to evaluate (required for most metrics),
 *   context?: string | string[] - retrieved context for evaluation,
 *   expected_output?: string - expected answer for contextual metrics,
 *   messages?: array - list of {role, content} for conversation_completeness,
 *   metric?: string (optional, defaults to 'answer_relevancy')
 * }
 *
 * Response:
 * {
 *   query?: string,
 *   output?: string,
 *   context?: string[],
 *   expected_output?: string,
 *   messages?: array,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/eval-only",
  asyncHandler(async (req: Request, res: Response) => {
    const { query, output, context, retrieval_context, expected_output, metric, messages } = req.body;

    // Unify context and retrieval_context fields (retrieval_context is alias for backward compatibility)
    const unifiedContext = context || retrieval_context;

    // Validation
    const effectiveMetric = metric || "answer_relevancy";
    if (effectiveMetric === "contextual_recall" || effectiveMetric === "contextual_precision") {
      if (!expected_output) {
        return res.status(400).json({
          error: "Missing required field: expected_output for " + effectiveMetric
        });
      }
      if (!unifiedContext) {
        return res.status(400).json({
          error: "Missing required field: context (or retrieval_context) for " + effectiveMetric
        });
      }
    } else if (effectiveMetric === "pii_leakage") {
      if (!query) {
        return res.status(400).json({
          error: "Missing required field: query for pii_leakage"
        });
      }
      if (!output) {
        return res.status(400).json({
          error: "Missing required field: output for pii_leakage"
        });
      }
    } else if (effectiveMetric === "hallucination") {
      if (!query) {
        return res.status(400).json({
          error: "Missing required field: query for hallucination"
        });
      }
      if (!unifiedContext) {
        return res.status(400).json({
          error: "Missing required field: context (or retrieval_context) for hallucination"
        });
      }
      if (!output) {
        return res.status(400).json({
          error: "Missing required field: output for hallucination"
        });
      }
    } else if (effectiveMetric === "conversation_completeness") {
      if (!messages) {
        return res.status(400).json({
          error: "Missing required field: messages for conversation_completeness"
        });
      }
    } else {
      if (!output) {
        return res.status(400).json({
          error: "Missing required field: output"
        });
      }
    }

    // Build evaluation parameters based on what's provided
    const evalParams: any = {
      metric: effectiveMetric,
      provider: "groq"
    };

    // Always add available fields regardless of metric type
    // This allows metric="all" to work with all necessary fields for each metric
    if (output) evalParams.output = output;
    if (query) evalParams.query = query;
    if (unifiedContext) evalParams.retrieval_context = Array.isArray(unifiedContext) ? unifiedContext : [unifiedContext];
    if (expected_output) evalParams.expected_output = expected_output;
    if (messages) evalParams.messages = messages;

    console.log(`Direct evaluation - Metric: ${effectiveMetric}`);
    if (query) console.log(`Query: ${query}`);
    if (unifiedContext) console.log(`Context: ${Array.isArray(unifiedContext) ? unifiedContext.length + ' items' : unifiedContext.substring(0, 100) + '...'}`);
    if (expected_output) console.log(`Expected Output: ${expected_output.substring(0, 100)}...`);
    if (output) console.log(`Output: ${output.substring(0, 100)}...`);
    if (messages) console.log(`Messages: ${messages.length} turns`);

    // Evaluate using specified metric (no LLM generation needed)
    const evalResult = await evalWithFields(evalParams);

    const response: any = {
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation,
        // Include results array if available for multi-metric support
        ...(evalResult.results && { results: evalResult.results })
      }
    };

    // Include optional fields in response if they were provided
    if (output) response.output = output;
    if (query) response.query = query;
    if (unifiedContext) response.context = Array.isArray(unifiedContext) ? unifiedContext : [unifiedContext];
    if (expected_output) response.expected_output = expected_output;
    if (messages) response.messages = messages;

    res.json(response);
  })
);

/**
 * GET /metrics
 * Get available evaluation metrics for training
 */
router.get("/metrics", async (req: Request, res: Response) => {
  try {
    // Fetch metrics info from Deepeval service
    const response = await fetch(`${ENV.DEEPEVAL_URL.replace('/eval', '/metrics-info')}`);
    const metricsInfo = await response.json();
    
    res.json({
      ...metricsInfo,
      usage_examples: {
        faithfulness: "Measures alignment with provided context - ideal for RAG systems",
        answer_relevancy: "Measures how well the answer addresses the question - good for QA systems"
      }
    });
  } catch (error) {
    res.status(500).json({
      error: "Could not fetch metrics information",
      available_metrics: ["faithfulness", "answer_relevancy"]
    });
  }
});

export default router;
