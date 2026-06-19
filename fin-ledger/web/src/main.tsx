import React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'
import { Layout } from './ui/Layout'
import { Dashboard } from './routes/Dashboard'
import { Transactions } from './routes/Transactions'
import { Fees } from './routes/Fees'
import { Spend } from './routes/Spend'
import { Statements } from './routes/Statements'
import { Documents } from './routes/Documents'
import { Benefits } from './routes/Benefits'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'transactions', element: <Transactions /> },
      { path: 'fees', element: <Fees /> },
      { path: 'spend', element: <Spend /> },
      { path: 'statements', element: <Statements /> },
      { path: 'documents', element: <Documents /> },
      { path: 'benefits', element: <Benefits /> }
    ]
  }
])

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)
