import React from 'react'
import { syncNow, getSpend, getFees } from '../lib/api'

export function Dashboard() {
  const [syncing, setSyncing] = React.useState(false)
  const [spend, setSpend] = React.useState<{total:number;count:number;last_date:string|null}|null>(null)
  const [fees, setFees] = React.useState<{count:number;total:number}|null>(null)

  React.useEffect(() => {
    const y = new Date().getFullYear()
    getSpend('Amazon', `${y}-01-01`).then(setSpend)
    const since = new Date(Date.now()-1000*60*60*24*90).toISOString().slice(0,10)
    getFees(since).then(setFees)
  }, [])

  async function onSync() {
    setSyncing(true)
    try { await syncNow() } finally { setSyncing(false) }
  }

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="card p-4 col-span-2">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold">Overview</h2>
          <button className="btn" onClick={onSync} disabled={syncing}>{syncing ? 'Syncing…' : 'Sync now'}</button>
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 rounded bg-slate-900 border border-slate-800">
            <div className="text-slate-400 text-sm">Amazon spend YTD</div>
            <div className="text-2xl font-semibold">${spend?.total?.toFixed(2) ?? '0.00'}</div>
            <div className="text-slate-400 text-xs">{spend?.count ?? 0} transactions</div>
          </div>
          <div className="p-4 rounded bg-slate-900 border border-slate-800">
            <div className="text-slate-400 text-sm">Fees last 90d</div>
            <div className="text-2xl font-semibold">${fees?.total?.toFixed(2) ?? '0.00'}</div>
            <div className="text-slate-400 text-xs">{fees?.count ?? 0} items</div>
          </div>
          <div className="p-4 rounded bg-slate-900 border border-slate-800">
            <div className="text-slate-400 text-sm">Last Amazon purchase</div>
            <div className="text-2xl font-semibold">{spend?.last_date ?? '—'}</div>
          </div>
        </div>
      </div>
      <div className="card p-4">
        <h2 className="text-lg font-semibold mb-2">Quick actions</h2>
        <ul className="space-y-2 text-sm">
          <li>• Sync Plaid transactions</li>
          <li>• Fetch a statement</li>
          <li>• Ingest a document</li>
        </ul>
      </div>
    </div>
  )
}
