import React from 'react'
import { ingestDocument, searchDocuments } from '../lib/api'

export function Documents() {
  const [title, setTitle] = React.useState('Sample Terms')
  const [content, setContent] = React.useState('Annual fee applies. This is a sample document body for search.')
  const [query, setQuery] = React.useState('annual fee')
  const [results, setResults] = React.useState<any[]>([])

  async function ingest() {
    await ingestDocument(title, content, 'manual')
    await runSearch()
  }

  async function runSearch() {
    const { results } = await searchDocuments(query)
    setResults(results)
  }

  React.useEffect(() => { runSearch() }, [])

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="card p-4">
        <h2 className="text-lg font-semibold mb-3">Ingest document</h2>
        <div className="space-y-2">
          <input className="input w-full" placeholder="Title" value={title} onChange={e=>setTitle(e.target.value)} />
          <textarea className="input w-full h-40" placeholder="Content" value={content} onChange={e=>setContent(e.target.value)} />
          <button className="btn" onClick={ingest}>Ingest</button>
        </div>
      </div>
      <div className="card p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">Search</h2>
          <div className="flex items-center gap-2">
            <input className="input" placeholder="Query" value={query} onChange={e=>setQuery(e.target.value)} />
            <button className="btn" onClick={runSearch}>Search</button>
          </div>
        </div>
        <ul className="divide-y divide-slate-800">
          {results.map((r) => (
            <li key={r.document_id} className="py-2">
              <div className="font-medium">{r.title}</div>
              <div className="text-slate-400 text-sm truncate">{r.content}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
