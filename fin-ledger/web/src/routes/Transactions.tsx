import React from 'react'
import { api } from '../lib/api'

type Txn = {
  transaction_id: string
  date: string
  name: string
  merchant_name?: string
  amount: number
}

export function Transactions() {
  const [rows, setRows] = React.useState<Txn[]>([])

  React.useEffect(() => {
    api.get('/documents/search', { params: { q: '' } }) // noop to verify API
    // For simplicity, fetch from a debug endpoint using DuckDB SQL via backend later. For now, pull spend for Amazon as demo.
  }, [])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Transactions</h2>
        <div className="text-sm text-slate-400">Demo: use top metrics and spend page</div>
      </div>
      <p className="text-slate-300">Use the Spend page for focused queries. We can add a general-purpose query endpoint later.</p>
    </div>
  )
}
