import React from 'react'
import { getStatement } from '../lib/api'

export function Statements() {
  const [accountId, setAccountId] = React.useState('acc_mock_cc')
  const [period, setPeriod] = React.useState('2025-09')
  const [meta, setMeta] = React.useState<{account_id:string; period:string; file_path:string; sha256:string} | null>(null)

  async function fetchOne() {
    setMeta(await getStatement(accountId, period))
  }

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Statements</h2>
        <div className="flex items-center gap-2">
          <input className="input" placeholder="Account ID" value={accountId} onChange={e=>setAccountId(e.target.value)} />
          <input className="input" placeholder="YYYY-MM" value={period} onChange={e=>setPeriod(e.target.value)} />
          <button className="btn" onClick={fetchOne}>Fetch</button>
        </div>
      </div>
      {meta && (
        <div className="space-y-2">
          <div className="text-sm text-slate-300">Account: {meta.account_id} • Period: {meta.period}</div>
          <div className="text-sm text-slate-400">SHA-256: <code>{meta.sha256}</code></div>
          <div className="text-sm text-slate-400">File: {meta.file_path}</div>
        </div>
      )}
    </div>
  )
}
