with stg_financial_statements as (
    select * from {{ ref('stg_financial_statements') }}
),

stg_companies as (
    select * from {{ ref('stg_companies') }}
),

extracted_fields as (
    select
        fs.id as statement_id,
        fs.company_id,
        fs.statement_type,
        fs.fiscal_year,
        fs.fiscal_period,
        -- Extract common financial metrics from JSON data
        (fs.data->>'revenue')::numeric as revenue,
        (fs.data->>'net_income')::numeric as net_income,
        (fs.data->>'total_assets')::numeric as total_assets,
        (fs.data->>'total_liabilities')::numeric as total_liabilities,
        (fs.data->>'shareholders_equity')::numeric as shareholders_equity,
        fs.created_at
    from stg_financial_statements fs
),

final as (
    select
        ef.statement_id,
        ef.company_id,
        c.symbol,
        ef.statement_type,
        ef.fiscal_year,
        ef.fiscal_period,
        ef.revenue,
        ef.net_income,
        ef.total_assets,
        ef.total_liabilities,
        ef.shareholders_equity,
        ef.created_at
    from extracted_fields ef
    join stg_companies c on ef.company_id = c.id
)

select * from final
