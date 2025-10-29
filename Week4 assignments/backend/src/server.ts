import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import path from 'path'
import JiraClient from 'jira-client'
import { generateRouter } from './routes/generate'

// Load environment variables from root directory
const envPath = path.join(__dirname, '../../.env')
console.log(`Loading .env from: ${envPath}`)
dotenv.config({ path: envPath })

// Debug environment variables
console.log('Environment variables loaded:')
console.log(`PORT: ${process.env.PORT}`)
console.log(`CORS_ORIGIN: ${process.env.CORS_ORIGIN}`)
console.log(`groq_API_BASE: ${process.env.groq_API_BASE}`)
console.log(`groq_API_KEY: ${process.env.groq_API_KEY ? 'SET' : 'NOT SET'}`)
console.log(`groq_MODEL: ${process.env.groq_MODEL}`)

const app = express()
const PORT = process.env.PORT || 8080

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true
}))
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true }))

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() })
})

// Add Jira client initialization
const jiraClient = new JiraClient({
  protocol: 'https',
  host: process.env.JIRA_HOST || 'mathantestingdemo.atlassian.net',
  username: process.env.JIRA_USERNAME,
  password: process.env.JIRA_API_TOKEN,
  apiVersion: '3',
  strictSSL: true
});

// Add Jira endpoint
app.post('/api/jira/fetch-story', async (req, res) => {
  try {
    const { jiraStoryLink } = req.body;

    if (!jiraStoryLink) {
      res.status(400).json({ error: 'Jira story link is required' });
      return;
    }

    // Extract issue key from URL
    const match = jiraStoryLink.match(/\/browse\/([A-Z]+-\d+)$/);
    if (!match) {
      res.status(400).json({ error: 'Invalid Jira URL format' });
      return;
    }
    const issueKey = match[1];
    console.log('Extracted issue key:', issueKey);

    // Fetch story from Jira
    const issue = await jiraClient.findIssue(issueKey);
    
     // Parse description to separate acceptance criteria
    let description = '';
    let acceptanceCriteria = '';
    
    if (issue.fields.description) {
      const descContent = issue.fields.description;
      
      // Check if description is an Atlassian Document Format (ADF)
      if (typeof descContent === 'object' && descContent.content) {
        // Extract text from ADF format
        description = descContent.content
          .map((block: any) => {
            if (block.content) {
              return block.content
                .map((item: any) => item.text || '')
                .join('')
            }
            return '';
          })
          .join('\n')
          .trim();
          
        // Try to extract acceptance criteria - assuming it's under a heading
        const sections = descContent.content;
        let inAcceptanceCriteria = false;
        
        acceptanceCriteria = sections
          .reduce((acc: string[], block: any) => {
            // Check for "Acceptance Criteria" heading
            if (block.type === 'heading' && 
                block.content?.[0]?.text?.toLowerCase().includes('acceptance criteria')) {
              inAcceptanceCriteria = true;
              return acc;
            }
            // Check for next heading to end AC section
            if (block.type === 'heading' && inAcceptanceCriteria) {
              inAcceptanceCriteria = false;
            }
            // Collect content while in AC section
            if (inAcceptanceCriteria && block.content) {
              const text = block.content
                .map((item: any) => item.text || '')
                .join('')
                .trim();
              if (text) acc.push(text);
            }
            return acc;
          }, [])
          .join('\n');
      } else {
        // Plain text description
        description = descContent.toString();
      }
    }

    // If no AC section found, include full description as AC
    if (!acceptanceCriteria) {
      acceptanceCriteria = description;
      description = ''; // Clear description to avoid duplication
    }
    console.log('Processed story data:', {
  title: issue.fields.summary,
  description,
  acceptanceCriteria,
  additionalInfo: `Priority: ${issue.fields.priority?.name || 'N/A'}\nStatus: ${issue.fields.status?.name || 'N/A'}`
});
    
    res.json({
      storyTitle: issue.fields.summary,
      description: description,
      acceptanceCriteria: acceptanceCriteria, // Adjust field ID as needed
      additionalInfo: `Priority: ${issue.fields.priority?.name || 'N/A'}\nStatus: ${issue.fields.status?.name || 'N/A'}`
    });
    
  } catch (error) {
    console.error('Error fetching Jira story:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to fetch Jira story'
    });
  }
});

// API routes
app.use('/api/generate-tests', generateRouter)

// Error handling middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Error:', err.message)
  res.status(500).json({
    error: process.env.NODE_ENV === 'production' ? 'Internal server error' : err.message
  })
})

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Route not found'
  })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Backend server running on port ${PORT}`)
  console.log(`📡 API available at http://localhost:${PORT}/api`)
  console.log(`🔍 Health check at http://localhost:${PORT}/api/health`)
})