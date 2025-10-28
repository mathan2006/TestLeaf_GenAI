declare module 'jira-client' {
  type JiraClientOptions = {
    protocol?: string
    host: string
    username?: string
    password?: string
    apiVersion?: string | number
    strictSSL?: boolean
    [key: string]: unknown
  }

  class JiraClient {
    constructor(options: JiraClientOptions)
    findIssue(issueKey: string): Promise<any>
    // add other methods you use here if needed
  }

  export default JiraClient
}